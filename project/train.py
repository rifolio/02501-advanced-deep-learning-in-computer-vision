import os
import torch as t
from torch import nn

from rich.progress import Progress

from data.dataloaders import framevideostack_trainloader, framevideostack_testloader
from models.early_fusion import EarlyFusion
from utils.timer import timer
from utils.logger import get_logger, log_metrics
from config import settings

from metrics.classification import Accuracy
from metrics.tracker import MetricTracker
from metrics.collection import MetricCollection

from losses.loss import LossFunction

loss_function = LossFunction(nn.CrossEntropyLoss())
test_loss_function = LossFunction(nn.CrossEntropyLoss())

train_metrics = MetricCollection({
    'train_accuracy': Accuracy(),
    'train_avg_loss': loss_function
})

test_metrics = MetricCollection({
    'test_accuracy': Accuracy(),
    'test_avg_loss': test_loss_function
})
train_tracker = MetricTracker()
test_tracker = MetricTracker()


logger = get_logger(__name__)


train_loader = framevideostack_trainloader
test_loader = framevideostack_testloader

 
model = EarlyFusion(in_size=64).to(settings.device)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

save_path = 'results/early_fusion.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

@timer
def train_eval(model, epochs=10):

    with Progress(transient=True) as progress:
        task = progress.add_task("[red]Training...", total=epochs)
        task_train = progress.add_task("[green]Training epoch...", total=len(train_loader))
        task_val = progress.add_task("[blue]Validating epoch...", total=len(test_loader))

        #best = 0.0
        #best_ep = 0
        for ep in range(1, epochs+1):

            # ---- Train
            train_metrics.reset()
            model.train()
            for X, y in train_loader:
                X, y = X.to(settings.device), y.to(settings.device)

                optimizer.zero_grad()
                logits = model(X)
                loss = loss_function.update(logits, y)#criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_metrics.update(logits, y)
                progress.update(task_train, advance=1)


            train_results = train_metrics.compute()
            train_tracker.update(ep, **train_results)

            # ---- Test
            test_metrics.reset()
            model.eval()
            with t.no_grad():
                for X, y in test_loader:
                    X, y = X.to(settings.device), y.to(settings.device)
                    logits = model(X)

                    test_metrics.update(logits, y)

                    progress.update(task_val, advance=1)

            test_results = test_metrics.compute()
            test_tracker.update(ep, **test_results)
            log_metrics(ep, **train_results, **test_results)
            
            # ---- Save best
            #if test_acc > best:
            #    best = test_acc
            #    best_ep = ep
            #    t.save(model.state_dict(), save_path)

            progress.reset(task_train)
            progress.reset(task_val)
            progress.update(task, advance=1)
            #scheduler.step()

    #logger.info(f"Best test: {best:.2f}% (epoch {best_ep}) → saved to {save_path}")
    #return best

train_eval(model, epochs=100)

train_tracker.save_csv('results/train_results.csv')
test_tracker.save_csv('results/test_results.csv')

#logger.info(f"Final training results: {results}")

