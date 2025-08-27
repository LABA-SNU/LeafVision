class Trainer():
    def __init__(model, classifier, arch, epochs):
        self.model = model
        self.classifier = classifier
        self.arch = arch
        self.dataloader = dataloader
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self, num_classes, optimizer, scheduler):
        self.model.train()
        self.classifier.train()
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        all_outputs = []
        all_targets = []

        train_stats = []

        for epoch in range(self.epochs):

            for inp, target in self.dataloader:
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and samples
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

                # Compute accuracy
                _, pred = torch.max(output, dim=1)
                total_correct += pred.eq(target).sum().item()

                # Accumulate outputs and targets for F1-score computation
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())

            if scheduler is not None:
                scheduler.step()
            
            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples

            # Concatenate all outputs and targets
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            # Compute F1-score
            avg_f1 = utils.f1_score(all_outputs, all_targets, num_classes)

            # Log statistics
            train_stat = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'lr': optimizer.param_groups[0]["lr"],
                'train_acc1': avg_acc,
                'train_f1': avg_f1
            }

            train_stats.append(train_stat)

        return train_stats

    def train_linear(self, num_classes, optimizer, scheduler):
        self.model.eval()
        self.classifier.train()
        
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        all_outputs = []
        all_targets = []

        train_stats = []

        for epoch in range(self.epochs):

            for inp, target in self.dataloader:
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and samples
                total_loss += loss.item() * self.batch_size
                total_samples += self.batch_size

                # Compute accuracy
                _, pred = torch.max(output, dim=1)
                total_correct += pred.eq(target).sum().item()

                # Accumulate outputs and targets for F1-score computation
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())

            if scheduler is not None:
                scheduler.step()
            
            # Compute average loss and accuracy
            avg_loss = total_loss / total_samples
            avg_acc = 100.0 * total_correct / total_samples

            # Concatenate all outputs and targets
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            # Compute F1-score
            avg_f1 = utils.f1_score(all_outputs, all_targets, num_classes)

            # Log statistics
            train_stat = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'lr': optimizer.param_groups[0]["lr"],
                'train_acc1': avg_acc,
                'train_f1': avg_f1
            }

            train_stats.append(train_stat)
        
        return train_stats


    @torch.no_grad()
    def validate(self, num_classes):
        self.model.eval()
        self.classifier.eval()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inp, target in self.dataloader:
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
        avg_f1 = utils.f1_score(all_outputs, all_targets, num_classes)

        # Log statistics
        val_stat = {
            'val_loss': avg_loss,
            'val_acc1': avg_acc,
            'val_f1': avg_f1
        }
        return val_stat