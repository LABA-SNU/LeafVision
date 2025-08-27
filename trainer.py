import torch
from torch import nn
import logging
import utils

class Trainer:
    def __init__(self, model, classifier, arch, dataloader, epochs, batch_size, optimizer, scheduler, logger=None, log_interval=50):
        self.model = model
        self.classifier = classifier
        self.arch = arch
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.test_loader = test_dataloader
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger  # optional external logger (e.g., wandb.log)
        self.log_interval = log_interval

    def _log(self, data: dict):
        """Log helper. Uses external logger if provided; otherwise falls back to logging.info."""
        if callable(self.logger):
            try:
                self.logger(data)
                return
            except Exception:
                pass
        logging.info(" ".join(f"{k}={v}" for k, v in data.items()))

    def train(self):
        self.model.train()
        self.classifier.train()
        
        train_stats = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_samples = 0
            total_correct = 0
            all_outputs = []
            all_targets = []

            for step, (inp, target) in enumerate(self.train_loader, start=1):
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                batch_size = inp.size(0)

                # Forward pass
                if "vit" in self.arch:
                    intermediate_output = self.model.get_intermediate_layers(inp, 1)
                    output_features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    avg_pooled = torch.mean(intermediate_output[-1][:, 1:], dim=1)
                    output_features = torch.cat((output_features, avg_pooled), dim=-1)
                else:
                    output_features = self.model(inp)
                output = self.classifier(output_features)

                # Compute loss
                loss = nn.CrossEntropyLoss()(output, target)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch-level logging
                if step % self.log_interval == 0:
                    self._log({
                        'phase': 'train',
                        'epoch': epoch + 1,
                        'step': step,
                        'batch_loss': float(loss.item()),
                        'lr': self.optimizer.param_groups[0]['lr'],
                    })

                # Accumulate loss and samples
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

                # Compute accuracy
                _, pred = torch.max(output, dim=1)
                total_correct += pred.eq(target).sum().item()

                # Accumulate outputs and targets for F1-score computation
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())

            if self.scheduler is not None:
                self.scheduler.step()
            
            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples

            # Concatenate all outputs and targets
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            # Compute F1-score
            avg_f1 = utils.f1_score(all_outputs, all_targets, self.classifier.num_labels)

            # Log statistics
            train_stat = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'lr': self.optimizer.param_groups[0]["lr"],
                'train_acc1': avg_acc,
                'train_f1': avg_f1
            }
            self._log({'phase': 'train_epoch', **train_stat})
            train_stats.append(train_stat)

        return train_stats

    def train_linear(self):
        self.model.eval()
        self.classifier.train()
        
        train_stats = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_samples = 0
            total_correct = 0
            all_outputs = []
            all_targets = []

            for step, (inp, target) in enumerate(self.train_loader, start=1):
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                batch_size = inp.size(0)

                # Forward pass
                with torch.no_grad():
                    if "vit" in self.arch:
                        intermediate_output = self.model.get_intermediate_layers(inp, 1)
                        output_features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                        avg_pooled = torch.mean(intermediate_output[-1][:, 1:], dim=1)
                        output_features = torch.cat((output_features, avg_pooled), dim=-1)
                    else:
                        output_features = self.model(inp)
                output = self.classifier(output_features)

                # Compute loss
                loss = nn.CrossEntropyLoss()(output, target)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch-level logging
                if step % self.log_interval == 0:
                    self._log({
                        'phase': 'train_linear',
                        'epoch': epoch + 1,
                        'step': step,
                        'batch_loss': float(loss.item()),
                        'lr': self.optimizer.param_groups[0]['lr'],
                    })

                # Accumulate loss and samples
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

                # Compute accuracy
                _, pred = torch.max(output, dim=1)
                total_correct += pred.eq(target).sum().item()

                # Accumulate outputs and targets for F1-score computation
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())

            if self.scheduler is not None:
                self.scheduler.step()
            
            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples

            # Concatenate all outputs and targets
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            # Compute F1-score
            avg_f1 = utils.f1_score(all_outputs, all_targets, self.classifier.num_labels)

            # Log statistics
            train_stat = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'lr': self.optimizer.param_groups[0]["lr"],
                'train_acc1': avg_acc,
                'train_f1': avg_f1
            }
            self._log({'phase': 'train_linear_epoch', **train_stat})
            train_stats.append(train_stat)
        
        return train_stats


    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        self.classifier.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for step, (inp, target) in enumerate(dataloader, start=1):
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                batch_size = inp.size(0)

                # Forward pass
                if "vit" in self.arch:
                    intermediate_output = self.model.get_intermediate_layers(inp, 1)
                    output_features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                    avg_pooled = torch.mean(intermediate_output[-1][:, 1:], dim=1)
                    output_features = torch.cat((output_features, avg_pooled), dim=-1)
                else:
                    output_features = self.model(inp)
                output = self.classifier(output_features)

                # Compute loss
                loss = nn.CrossEntropyLoss()(output, target)

                # Accumulate loss and samples
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

                # Compute accuracy
                _, pred = torch.max(output, dim=1)
                total_correct += pred.eq(target).sum().item()

                # Accumulate outputs and targets for F1-score computation
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())

        # Compute average loss and accuracy
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples

        # Concatenate all outputs and targets
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Compute F1-score
        avg_f1 = utils.f1_score(all_outputs, all_targets, self.classifier.num_labels)

        # Log statistics
        val_stat = {
            'val_loss': avg_loss,
            'val_acc1': avg_acc,
            'val_f1': avg_f1
        }
        self._log({'phase': 'validate', **val_stat})
        return val_stat
    
    def train_