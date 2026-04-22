"""
GUI to test CIFAR-100 SNN on test-set samples or arbitrary user images.

Shows: input image, per-layer spike heatmaps, output raster, top-K bar chart,
and per-layer fire rate over time. Buttons: Next (cycle test set),
Load Image (open any .png/.jpg and resize to 32x32).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from snntorch import spikeplot as splt
    HAS_SPIKEPLOT = True
except ImportError:
    HAS_SPIKEPLOT = False

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

from cifar100.model import SNNConvClassifier
from cifar100.data import DataModuleCIFAR100, CIFAR100_MEAN, CIFAR100_STD


# --- Setup ---------------------------------------------------------------

EXPERIMENT_NAME = os.environ.get('CIFAR100_GUI_EXP', 'cifar100_p3_Q')
TOP_K = 8
beta = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = PROJECT_ROOT / 'checkpoints' / f'{EXPERIMENT_NAME}_best.pth'
if not checkpoint_path.exists():
    print(f"Error: checkpoint not found at {checkpoint_path}")
    print("Set CIFAR100_GUI_EXP env var to another experiment if needed.")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
classes = checkpoint['classes']
num_classes = len(classes)
num_steps = checkpoint.get('num_steps', 8)
bn_mode = checkpoint.get('bn_mode', 'bntt')

net = SNNConvClassifier(
    num_classes=num_classes,
    num_steps=num_steps,
    beta=beta,
    bn_mode=bn_mode,
).to(device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
print(f"Loaded {EXPERIMENT_NAME} epoch {checkpoint['epoch']} "
      f"(accuracy: {checkpoint['accuracy']:.2f}%)")
print(f"snntorch.spikeplot: {'available' if HAS_SPIKEPLOT else 'fallback'}")

# --- Data ----------------------------------------------------------------

data_module = DataModuleCIFAR100(
    batch_size=1,
    data_path=str(PROJECT_ROOT / 'data' / 'cifar-100'),
    aug_level='none',
)
_, test_loader = data_module.get_dataloaders()

mean_t = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
std_t = torch.tensor(CIFAR100_STD).view(3, 1, 1)

custom_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


def denormalize(img):
    """[3,H,W] normalized tensor -> [H,W,3] uint8 for display."""
    img = img.cpu() * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


class InferenceGUI:
    def __init__(self):
        self.test_iter = iter(test_loader)
        self.current_img = None   # normalized [1, 3, 32, 32]
        self.current_target = None
        self.current_label = ''

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'CIFAR-100 SNN — {EXPERIMENT_NAME}',
                          fontsize=15, fontweight='bold')
        gs = GridSpec(3, 4, figure=self.fig,
                      hspace=0.55, wspace=0.35,
                      left=0.05, right=0.97, top=0.90, bottom=0.12)

        self.ax_input = self.fig.add_subplot(gs[0, 0])
        self.ax_lif1 = self.fig.add_subplot(gs[0, 1])
        self.ax_lif2 = self.fig.add_subplot(gs[0, 2])
        self.ax_lif3 = self.fig.add_subplot(gs[0, 3])

        self.ax_lif4 = self.fig.add_subplot(gs[1, 0])
        self.ax_raster = self.fig.add_subplot(gs[1, 1:3])
        self.ax_topk = self.fig.add_subplot(gs[1, 3])

        self.ax_fire = self.fig.add_subplot(gs[2, :])

        self.title_txt = self.fig.text(0.5, 0.94, '', ha='center',
                                       fontsize=12, fontweight='bold')

        ax_next = self.fig.add_axes([0.74, 0.02, 0.10, 0.05])
        self.btn_next = Button(ax_next, 'Next (test)')
        self.btn_next.on_clicked(self.on_next)

        ax_load = self.fig.add_axes([0.86, 0.02, 0.11, 0.05])
        self.btn_load = Button(ax_load, 'Load Image')
        self.btn_load.on_clicked(self.on_load)

        self.load_next_sample()

    # ---- sample loaders ------------------------------------------------

    def load_next_sample(self):
        try:
            img, target = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(test_loader)
            img, target = next(self.test_iter)
        self.current_img = img.to(device)
        self.current_target = target.item()
        self.current_label = f"True: {classes[self.current_target]}"
        self.run_inference()

    def load_test_index(self, idx):
        img, target = test_loader.dataset[idx]
        self.current_img = img.unsqueeze(0).to(device)
        self.current_target = int(target)
        self.current_label = f"Test idx {idx} — True: {classes[self.current_target]}"
        self.run_inference()

    def load_custom_image(self, path):
        img = Image.open(path).convert('RGB')
        tensor = custom_transform(img).unsqueeze(0).to(device)
        self.current_img = tensor
        self.current_target = None
        self.current_label = f"Custom: {os.path.basename(path)}"
        self.run_inference()

    # ---- inference ----------------------------------------------------

    def run_inference(self):
        with torch.no_grad():
            spk_out_rec, mem_out, fire_rates, logit_rec, recs = \
                self.net_forward(self.current_img)
            logits = logit_rec.mean(dim=0)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        self._draw_input()
        self._draw_spatial(self.ax_lif1, recs['lif1'], 'LIF1 spikes (Σ T,C)')
        self._draw_spatial(self.ax_lif2, recs['lif2'], 'LIF2 spikes (Σ T,C)')
        self._draw_spatial(self.ax_lif3, recs['lif3'], 'LIF3 spikes (Σ T,C)')
        self._draw_spatial(self.ax_lif4, recs['lif4'], 'LIF4 spikes (Σ T,C)')
        self._draw_logit_heatmap(logit_rec[:, 0], logits[0], pred)
        self._draw_topk(logits[0], pred)
        self._draw_fire_rates(recs)

        pred_class = classes[pred]
        mark = ''
        if self.current_target is not None:
            mark = '  [correct]' if pred == self.current_target else '  [wrong]'
        layer_fire = " / ".join(f"{fr.item():.3f}" for fr in fire_rates)
        self.title_txt.set_text(
            f"{self.current_label}  |  Pred: {pred_class} ({conf*100:.1f}%){mark}  "
            f"|  fire lif1/2/3/4/out = {layer_fire}"
        )
        self.fig.canvas.draw_idle()

    def net_forward(self, img):
        return net(img, record_all=True)

    # ---- drawing ------------------------------------------------------

    def _draw_input(self):
        self.ax_input.clear()
        self.ax_input.imshow(denormalize(self.current_img[0]))
        self.ax_input.set_title('Input image (32x32)', fontsize=10)
        self.ax_input.axis('off')

    def _draw_spatial(self, ax, spk_layer, title):
        ax.clear()
        heatmap = spk_layer[:, 0].sum(dim=(0, 1)).cpu().numpy()
        ax.imshow(heatmap, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def _draw_logit_heatmap(self, logit_rec_t, logits, pred):
        """Heatmap of per-timestep logits for top-K classes.

        Output LIF rarely fires under TET+logit-readout training (logits stay
        below threshold), so a spike raster would be empty. The pre-LIF fc
        output is what the classifier actually uses — show its evolution here.
        """
        self.ax_raster.clear()
        top_k_idx = torch.topk(logits, TOP_K).indices.cpu().numpy()
        if self.current_target is not None and self.current_target not in top_k_idx:
            top_k_idx = np.concatenate([top_k_idx, [self.current_target]])

        # [len(top_k_idx), T] for imshow (classes on y, time on x)
        heatmap = logit_rec_t[:, top_k_idx].cpu().numpy().T
        self.ax_raster.imshow(heatmap, aspect='auto', cmap='viridis',
                              origin='lower')

        self.ax_raster.set_yticks(range(len(top_k_idx)))
        labels = []
        for ci in top_k_idx:
            name = classes[ci]
            if self.current_target is not None and ci == self.current_target and ci == pred:
                name = f"* {name} *"
            elif self.current_target is not None and ci == self.current_target:
                name = f"[T] {name}"
            elif ci == pred:
                name = f"[P] {name}"
            labels.append(name)
        self.ax_raster.set_yticklabels(labels, fontsize=8)
        self.ax_raster.set_xlabel('time step')
        self.ax_raster.set_title(
            f'Logit evolution — top-{TOP_K} classes (pre-LIF fc output)',
            fontsize=10)

    def _draw_topk(self, logits, pred):
        self.ax_topk.clear()
        top_vals, top_idx = torch.topk(logits, TOP_K)
        top_vals = top_vals.cpu().numpy()
        top_idx = top_idx.cpu().numpy()
        names = [classes[i] for i in top_idx]
        colors = []
        for i in top_idx:
            if self.current_target is not None and i == self.current_target:
                colors.append('red')
            elif i == pred:
                colors.append('green')
            else:
                colors.append('steelblue')
        y = np.arange(len(names))
        self.ax_topk.barh(y, top_vals, color=colors)
        self.ax_topk.set_yticks(y)
        self.ax_topk.set_yticklabels(names, fontsize=8)
        self.ax_topk.invert_yaxis()
        self.ax_topk.set_xlabel('mean logit (TET readout)')
        self.ax_topk.set_title(f'Top-{TOP_K} classes\n(red=true, green=pred)',
                               fontsize=10)

    def _draw_fire_rates(self, recs):
        self.ax_fire.clear()
        for name in ('lif1', 'lif2', 'lif3', 'lif4', 'out'):
            rec = recs[name]
            fr = rec[:, 0].float()
            fr = fr.reshape(fr.shape[0], -1).mean(dim=1).cpu().numpy()
            self.ax_fire.plot(fr, label=name, linewidth=1.5)
        self.ax_fire.set_xlabel('time step')
        self.ax_fire.set_ylabel('mean firing rate')
        self.ax_fire.set_title('Per-layer firing rate over time', fontsize=10)
        self.ax_fire.legend(loc='upper right', fontsize=9)
        self.ax_fire.grid(True, alpha=0.3)
        self.ax_fire.set_xlim(0, num_steps - 1)

    # ---- buttons ------------------------------------------------------

    def on_next(self, _event):
        self.load_next_sample()

    def on_load(self, _event):
        if not HAS_TK:
            print("tkinter not available — cannot open file dialog.")
            return
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='Choose image',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp')],
        )
        root.destroy()
        if path:
            self.load_custom_image(path)


def run_once_and_save(gui, out_path):
    """Render current sample to a PNG (no display needed)."""
    gui.fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved figure -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=None,
                        help='run on test sample at this index, save PNG')
    parser.add_argument('--image', type=str, default=None,
                        help='run on this image file, save PNG')
    parser.add_argument('--output', type=str, default=None,
                        help='output PNG path (default auto-named in viz/)')
    parser.add_argument('--force-gui', action='store_true',
                        help='try interactive GUI even if DISPLAY missing')
    args = parser.parse_args()

    headless = (not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'))
    save_mode = args.index is not None or args.image is not None

    if headless and not args.force_gui and not save_mode:
        print("Headless environment detected (no DISPLAY).")
        print("Use --index N, --image PATH, or set --force-gui.")
        print("Example: python cifar100/inference_gui.py --index 42")
        sys.exit(0)

    if save_mode:
        matplotlib.use('Agg')
        gui = InferenceGUI()
        viz_dir = PROJECT_ROOT / 'viz'
        viz_dir.mkdir(exist_ok=True)

        if args.image is not None:
            gui.load_custom_image(args.image)
            default_name = Path(args.image).stem + '_prediction.png'
        else:
            gui.load_test_index(args.index)
            default_name = f'test_{args.index:05d}.png'

        out = Path(args.output) if args.output else (viz_dir / default_name)
        run_once_and_save(gui, out)
    else:
        gui = InferenceGUI()
        plt.show()
