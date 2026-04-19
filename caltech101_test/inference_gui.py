"""
Simple GUI to test the 5-class N-Caltech101 model on test set samples.
Shows predicted class + confidence, click "Next" to see next sample.
"""

import sys
from pathlib import Path

from snntorch import surrogate
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from caltech101_test.data import DataModuleNCaltech101Test
from models.snn import SNNClassifier

spike_grad = surrogate.fast_sigmoid()

EXPERIMENT_NAME = "caltech101_test_fast_sigmoid"
SEED = 42
num_steps = 100
num_inputs = 2 * 180 * 240
num_hidden = 1000
beta = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = PROJECT_ROOT / "checkpoints" / f"{EXPERIMENT_NAME}_best.pth"
if not checkpoint_path.exists():
    print(f"Error: checkpoint not found at {checkpoint_path}")
    sys.exit(1)

data_module = DataModuleNCaltech101Test(batch_size=1, num_steps=num_steps, seed=SEED)
_, test_loader = data_module.get_dataloaders()
classes = data_module.selected_classes
num_classes = len(classes)

net = SNNClassifier(
    num_inputs,
    num_hidden,
    num_classes,
    num_steps,
    beta,
    spike_grad=spike_grad,
).to(device)
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint["model_state_dict"])
net.eval()
print(f"Loaded model from epoch {checkpoint['epoch']} (accuracy: {checkpoint['accuracy']:.2f}%)")
print(f"Inference classes: {classes}")
print(f"Experiment: {EXPERIMENT_NAME} | Seed: {SEED}")


class InferenceGUI:
    def __init__(self, test_loader, net, classes, device):
        self.test_loader = test_loader
        self.net = net
        self.classes = classes
        self.device = device
        self.test_iter = iter(test_loader)
        self.current_events = None
        self.current_target = None

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle("N-Caltech101 5-Class Inference", fontsize=16, fontweight="bold")

        ax_next = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.btn_next = Button(ax_next, "Next")
        self.btn_next.on_clicked(self.on_next)

        self.load_next_sample()

    def load_next_sample(self):
        try:
            events, target = next(self.test_iter)
        except StopIteration:
            print("End of test set reached. Restarting...")
            self.test_iter = iter(self.test_loader)
            events, target = next(self.test_iter)

        self.current_events = events.float().to(self.device)
        self.current_target = target.item()
        self.run_inference()

    def run_inference(self):
        with torch.no_grad():
            events = self.current_events.transpose(0, 1)
            events_flat = events.reshape(num_steps, 1, -1)

            spk_rec, _ = self.net(None, x_seq=events_flat)

            logits = spk_rec.sum(dim=0)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()

        self.ax.clear()

        events_vis = self.current_events[0].sum(dim=0).sum(dim=0).cpu().numpy()

        self.ax.imshow(events_vis, cmap="hot")
        self.ax.axis("off")

        true_class = self.classes[self.current_target]
        pred_class = self.classes[pred]
        is_correct = "correct" if pred == self.current_target else "wrong"

        title = (
            f"True: {true_class}\n"
            f"Pred: {pred_class} ({conf * 100:.1f}%) [{is_correct}]"
        )
        self.ax.set_title(title, fontsize=12, fontweight="bold")

        self.fig.canvas.draw_idle()

    def on_next(self, event):
        self.load_next_sample()


gui = InferenceGUI(test_loader, net, classes, device)
plt.show()
