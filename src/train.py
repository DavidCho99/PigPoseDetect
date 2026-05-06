import copy

'''
This code from the example from Section 13.2
'''


def train_model(model, train_loader, val_loader, min_epochs, max_epochs, patience=5):
    train_loss, train_acc = model.assess(train_loader)
    val_loss, val_acc = model.assess(val_loader)

    results = {'train_loss': [train_loss], 'train_acc': [train_acc],
               'val_loss': [val_loss], 'val_acc': [val_acc]}

    print(f"Prior to training, results={results}")
    epoch = 0
    epochs_since_improvement = 0
    best_val_acc = val_acc
    best_val_loss = val_loss
    training = True

    while training:
        epoch += 1
        print(f"*** Epoch {epoch}/{max_epochs} ***")
        train_loss, train_acc = model.train_epoch(train_loader)
        val_loss, val_acc = model.assess(val_loader)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        # we find the best validation metric.
        if val_acc > best_val_acc or val_loss < best_val_loss:
            best_val_acc = max(val_acc, best_val_acc)
            best_val_loss = min(val_loss, best_val_loss)
            epochs_since_improvement = 0
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improvement += 1
        print(f"Train: loss={train_loss:0.4f}, acc={train_acc:0.4f}")
        print(f"  Val: loss={val_loss:0.4f}, acc={val_acc:0.4f}")
        print(f" best:      {best_val_loss:0.4f},     {best_val_acc:0.4f}, " +
              f"{epochs_since_improvement} epochs since improved")

        # Are we done training?
        if epoch >= min_epochs:
            if epoch >= max_epochs or epochs_since_improvement >= patience:
                training = False

    # Restore the state at the last improvement.
    print(f"Restoring model to state after epoch {best_epoch}.")
    model.load_state_dict(best_weights)
    model.eval()
    results['epoch_used'] = best_epoch
    # return the dictionary for the result
    return results
