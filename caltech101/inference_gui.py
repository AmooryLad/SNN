"""
GUI to test N-Caltech101 model on test set samples with spike visualizations.
Shows input events, per-layer spike activity, output raster, and top-K bar chart.
Click "Next" to see next sample.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from snntorch import spikeplot as splt
    HAS_SPIKEPLOT = True
except ImportError:
    HAS_SPIKEPLOT = False

from caltech101.model import SNNConvClassifier
from caltech101.data import DataModuleNCaltech101

# --- Setup ---

EXPERIMENT_NAME = "caltech101_conv_snn"
beta = 0.95
TOP_K = 8  # number of top predicted classes to show in raster/bar chart

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---

checkpoint_path = f'../checkpoints/{EXPERIMENT_NAME}_best.pth'
if not os.path.exists(checkpoint_path):
    print(f"Error: checkpoint not found at {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
classes = checkpoint['classes']
num_classes = len(classes)
num_steps = checkpoint.get('num_steps', 50)

net = SNNConvClassifier(
    num_classes=num_classes,
    num_steps=num_steps,
    beta=beta,
).to(device)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
print(f"Loaded model from epoch {checkpoint['epoch']} (accuracy: {checkpoint['accuracy']:.2f}%)")
print(f"snntorch.spikeplot: {'available' if HAS_SPIKEPLOT else 'not available, using matplotlib fallback'}")

# --- Load data ---

data_module = DataModuleNCaltech101(
    batch_size=1,
    data_path='../data/ncaltech101/caltech101',
    num_steps=num_steps,
)
_, test_loader = data_module.get_dataloaders(subset=10)


class InferenceGUI:
    def __init__(self, test_loader, net, classes, device):
        self.test_loader = test_loader
        self.net = net
        self.classes = classes
        self.device = device
        self.test_iter = iter(test_loader)

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('N-Caltech101 SNN — Spike Visualization',
                          fontsize=15, fontweight='bold')
        gs = GridSpec(3, 3, figure=self.fig,
                      hspace=0.55, wspace=0.35,
                      left=0.05, right=0.97, top=0.90, bottom=0.10)

        # Row 0: input event heatmap + spatial heatmaps of conv layers
        self.ax_input = self.fig.add_subplot(gs[0, 0])
        self.ax_lif1 = self.fig.add_subplot(gs[0, 1])
        self.ax_lif2 = self.fig.add_subplot(gs[0, 2])

        # Row 1: lif3 spatial, output raster, top-K bar chart
        self.ax_lif3 = self.fig.add_subplot(gs[1, 0])
        self.ax_raster = self.fig.add_subplot(gs[1, 1])
        self.ax_topk = self.fig.add_subplot(gs[1, 2])

        # Row 2: per-layer fire rate over time (spans all 3 cols)
        self.ax_fire = self.fig.add_subplot(gs[2, :])

        # Title area for prediction text
        self.title_txt = self.fig.text(0.5, 0.94, '', ha='center',
                                       fontsize=12, fontweight='bold')

        # Next button
        ax_next = self.fig.add_axes([0.87, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.on_next)

        self.load_next_sample()

    def load_next_sample(self):
        try:
            events, target = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_loader)
            events, target = next(self.test_iter)

        self.current_events = events.float().to(self.device)
        self.current_target = target.item()
        self.run_inference()

    def run_inference(self):
        with torch.no_grad():
            # [B=1, T, 2, H, W] -> [T, B, 2, H, W]
            events = self.current_events.transpose(0, 1)
            spk_out_rec, fire_rates, recs = self.net(events, record_all=True)
            logits = spk_out_rec.sum(dim=0)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        self._draw_input()
        self._draw_spatial(self.ax_lif1, recs['lif1'], 'LIF1 spatial spikes (Σ over T, channels)')
        self._draw_spatial(self.ax_lif2, recs['lif2'], 'LIF2 spatial spikes (Σ over T, channels)')
        self._draw_spatial(self.ax_lif3, recs['lif3'], 'LIF3 spatial spikes (Σ over T, channels)')
        self._draw_raster(recs['out'][:, 0], logits[0], pred)
        self._draw_topk(logits[0], pred)
        self._draw_fire_rates(recs)

        true_class = self.classes[self.current_target]
        pred_class = self.classes[pred]
        mark = 'correct' if pred == self.current_target else 'wrong'
        layer_fire = " / ".join(f"{fr.item():.3f}" for fr in fire_rates)
        self.title_txt.set_text(
            f"True: {true_class}  |  Pred: {pred_class} ({conf*100:.1f}%)  [{mark}]  "
            f"| avg fire lif1/lif2/lif3/out = {layer_fire}"
        )
        self.fig.canvas.draw_idle()

    def _draw_input(self):
        self.ax_input.clear()
        events_vis = self.current_events[0].sum(dim=0).sum(dim=0).cpu().numpy()
        self.ax_input.imshow(events_vis, cmap='hot', aspect='auto')
        self.ax_input.set_title('Input events (Σ over T, polarity)', fontsize=10)
        self.ax_input.axis('off')

    def _draw_spatial(self, ax, spk_layer, title):
        """Heatmap of spike counts summed over time and channels.
        spk_layer: [T, B=1, C, H, W]
        """
        ax.clear()
        heatmap = spk_layer[:, 0].sum(dim=(0, 1)).cpu().numpy()
        ax.imshow(heatmap, cmap='viridis', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    def _draw_raster(self, out_spikes, logits, pred):
        """Raster plot of output layer spikes over time.
        out_spikes: [T, num_classes]
        Only shows top-K predicted classes + the true class for readability.
        """
        self.ax_raster.clear()
        top_k_idx = torch.topk(logits, TOP_K).indices.cpu().numpy()
        if self.current_target not in top_k_idx:
            top_k_idx = np.concatenate([top_k_idx, [self.current_target]])

        # Build subset raster: rows = selected classes, cols = time steps
        spk_sub = out_spikes[:, top_k_idx].cpu().numpy()  # [T, K]

        if HAS_SPIKEPLOT:
            # spikeplot.raster expects [T, N]
            splt.raster(torch.tensor(spk_sub), self.ax_raster, s=20, c='black')
        else:
            # Fallback: scatter points where spikes happened
            t_idx, n_idx = np.where(spk_sub > 0)
            self.ax_raster.scatter(t_idx, n_idx, s=20, c='black', marker='|')

        self.ax_raster.set_yticks(range(len(top_k_idx)))
        labels = []
        for ci in top_k_idx:
            name = self.classes[ci]
            if ci == self.current_target and ci == pred:
                name = f"* {name} *"
            elif ci == self.current_target:
                name = f"[T] {name}"
            elif ci == pred:
                name = f"[P] {name}"
            labels.append(name)
        self.ax_raster.set_yticklabels(labels, fontsize=8)
        self.ax_raster.set_xlabel('time step')
        self.ax_raster.set_title(f'Output raster — top-{TOP_K} classes + true', fontsize=10)
        self.ax_raster.set_xlim(-0.5, num_steps - 0.5)
        self.ax_raster.grid(True, alpha=0.2)

    def _draw_topk(self, logits, pred):
        """Bar chart of top-K class spike counts."""
        self.ax_topk.clear()
        top_vals, top_idx = torch.topk(logits, TOP_K)
        top_vals = top_vals.cpu().numpy()
        top_idx = top_idx.cpu().numpy()
        names = [self.classes[i] for i in top_idx]
        colors = ['red' if i == self.current_target else
                  ('green' if i == pred else 'steelblue') for i in top_idx]
        y = np.arange(len(names))
        self.ax_topk.barh(y, top_vals, color=colors)
        self.ax_topk.set_yticks(y)
        self.ax_topk.set_yticklabels(names, fontsize=8)
        self.ax_topk.invert_yaxis()
        self.ax_topk.set_xlabel('accumulated spikes')
        self.ax_topk.set_title(f'Top-{TOP_K} class spike counts\n(red=true, green=pred)', fontsize=10)

    def _draw_fire_rates(self, recs):
        """Mean firing rate of each layer across time steps."""
        self.ax_fire.clear()
        for name, rec in (('lif1', recs['lif1']), ('lif2', recs['lif2']),
                          ('lif3', recs['lif3']), ('out', recs['out'])):
            # rec: [T, B, ...] — mean over all non-time dims
            fr = rec[:, 0].float()
            fr = fr.reshape(fr.shape[0], -1).mean(dim=1).cpu().numpy()
            self.ax_fire.plot(fr, label=name, linewidth=1.5)
        self.ax_fire.set_xlabel('time step')
        self.ax_fire.set_ylabel('mean firing rate')
        self.ax_fire.set_title('Per-layer firing rate over time (sparsity = energy efficiency)',
                               fontsize=10)
        self.ax_fire.legend(loc='upper right', fontsize=9)
        self.ax_fire.grid(True, alpha=0.3)
        self.ax_fire.set_xlim(0, num_steps - 1)

    def on_next(self, event):
        self.load_next_sample()


gui = InferenceGUI(test_loader, net, classes, device)
plt.show()
