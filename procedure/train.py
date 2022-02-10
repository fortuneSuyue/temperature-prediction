import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BasicTrainer:
    def __init__(self, loss_func=None, cuda=False):
        self.loss_func = loss_func
        self.cuda = cuda

    def train_per_epoch(self, dataloader: DataLoader, model, opt, loss_func=None, evaluation: dict = None, cuda=None,
                        info='', is_seq2seq=False):
        if cuda is not None and isinstance(cuda, bool):
            self.cuda = cuda
        if self.cuda:
            model.cuda()
        else:
            model.cpu()
        if loss_func is not None:
            self.loss_func = loss_func
        grad_loss = 0.
        if evaluation is not None:
            evaluation_values = {key: 0. for key in evaluation.keys()}
        else:
            evaluation_values = None
        model.train()
        num_batch = len(dataloader)
        tbar = tqdm(dataloader)
        for i, (train_x, train_y) in enumerate(tbar):
            if self.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()
            if is_seq2seq:
                pred_y = model(train_x, train_y)
            else:
                pred_y = model(train_x)
            loss = self.loss_func(pred_y, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            grad_loss += loss.item()
            if evaluation is not None:
                for key in evaluation.keys():
                    evaluation_values[key] += evaluation[key](pred_y, train_y).item()
            tbar.set_description(desc=f'Training {info}  [{i + 1}/{num_batch}]: {grad_loss / (i + 1)}')
        grad_loss /= num_batch
        if evaluation is not None:
            for key in evaluation_values.keys():
                evaluation_values[key] /= num_batch
            return grad_loss, evaluation_values
        return grad_loss

    def validation(self, dataloader: DataLoader, model, loss_func=None, evaluation: dict = None, cuda=None,
                   is_transformer=False):
        if cuda is not None:
            if cuda:
                model.cuda()
            else:
                model.cpu()
        if loss_func is None:
            loss_func = self.loss_func
        assert loss_func is not None, 'loss_func is None and BasicTrainer.loss_func is None...'
        common_loss = 0.
        if evaluation is not None:
            evaluation_values = {key: 0. for key in evaluation.keys()}
        else:
            evaluation_values = None
        model.eval()
        num_batch = len(dataloader)
        tbar = tqdm(dataloader)
        for i, (train_x, train_y) in enumerate(tbar):
            if self.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()
            with torch.no_grad():
                if is_transformer:
                    pred_y = model(train_x, train_y)
                else:
                    pred_y = model(train_x)
            common_loss += loss_func(pred_y, train_y).item()
            if evaluation is not None:
                for key in evaluation.keys():
                    evaluation_values[key] += evaluation[key](pred_y, train_y).item()
            tbar.set_description(desc=f'Validation [{i + 1}/{num_batch}]: {common_loss / (i + 1)}')
        common_loss /= num_batch
        if evaluation is not None:
            for key in evaluation_values.keys():
                evaluation_values[key] /= num_batch
            print(evaluation_values)
            return common_loss, evaluation_values
        return common_loss

    def cuda(self):
        self.cuda = True
