import math
from dataset import *
from sklearn.metrics import mean_squared_error


def my_accuracy(true_rul, pred_rul):
    diff = (pred_rul - true_rul)
    for i in range(0, len(diff)):
        if 10 >= diff[i] >= -13:
            diff[i] = 1
        else:
            diff[i] = 0
    return torch.sum(diff)


def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return torch.sum(torch.where(diff < 0, torch.exp(-diff/13)-1, torch.exp(diff/10)-1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def testing_function(model, test_loader, loss_func,  max_rul, device):
    model.eval()
    s_score, score, accuracy, rmse_test = 0, 0, 0, 0

    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        with torch.no_grad():
            test_predict, _ = model.forward(x_test)
            scale_test_predict = (test_predict*max_rul).floor()
            scale_y_test = y_test * max_rul
            rmse_test += loss_func(scale_test_predict, scale_y_test).item()
            s_score += compute_s_score(scale_y_test, scale_test_predict)
            accuracy += my_accuracy(scale_y_test, scale_test_predict)

    test_predict = test_predict.cpu().numpy()
    return rmse_test, s_score, accuracy, test_predict


def test_five_sample(model, test_loader, loss_func, max_rul, num_test_windows, device):
    model.eval()
    score, accuracy, rmse_test = 0, 0, 0
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        with torch.no_grad():
            y_test = (y_test.reshape(-1)) * max_rul
            rul_pred, attention_weight_test = model.forward(x_test)
            rul_pred = rul_pred.reshape(-1)
            rul_pred = rul_pred.detach()
            rul_pred *= max_rul
            preds_for_each_engine = torch.split(rul_pred, num_test_windows)  # [array, array,..]
            y_test = torch.split(y_test, num_test_windows)
            y_test = torch.tensor([item.sum() / len(item) for item in y_test])
            mean_pred_for_each_engine = torch.tensor([item.sum() / len(item) for item in preds_for_each_engine])
            rmse_test += loss_func(y_test, mean_pred_for_each_engine).item()
            score += compute_s_score(y_test, mean_pred_for_each_engine)
            accuracy += my_accuracy(y_test, mean_pred_for_each_engine)

    return rmse_test, score.item()


def validating_function(model, valid_loader, loss_func, max_rul, device):
    rmse_valid = 0
    valid_batch_len = len(valid_loader)
    size = len(valid_loader.dataset)
    score = 0
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            valid_predict, attention_weight_valid = model(x_valid)
            rmse_valid += loss_func(valid_predict, y_valid).item()
            score += compute_s_score(y_valid*125, valid_predict*125)
    rmse_valid /= valid_batch_len
    rmse_valid *= max_rul
    return rmse_valid, score.item(), attention_weight_valid


def train(model_for_train, train_loader, valid_loader, test_loader, N_EPOCH, optimizer, scheduler, loss_train,
          loss_eval, lines_list, patience, max_rul, num_test_windows, device):
    train_batch_length = len(train_loader)
    for epoch in range(1, N_EPOCH + 1):
        model_for_train.train()
        model_for_train.to(device)
        epoch_loss = 0

        for batch, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            outputs = model_for_train(x_train)  # forward pass
            loss = loss_train(outputs[0], y_train)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_for_train.eval()
        rmse_valid, valid_score, attention_weight_valid = validating_function(model_for_train,
                                                                              valid_loader=valid_loader,
                                                                              loss_func=loss_eval,
                                                                              max_rul=max_rul,
                                                                              device=device)
        rmse_test, score_test = test_five_sample(model_for_train, test_loader, loss_eval, max_rul, num_test_windows, device)
        epoch_loss /= train_batch_length
        epoch_loss = math.sqrt((epoch_loss)) * max_rul
        content = "Epoch: %d, Loss: %1.4f, Rmse_valid: %1.4f, valid_score: %1.1f, " \
                  ", rmse_test: %1.4f, score_test: %1.1f, Learning rate:%1.5f" % \
                  (epoch, epoch_loss, rmse_valid, valid_score, rmse_test, score_test, optimizer.state_dict()['param_groups'][0]['lr'])
        print(content)
        scheduler.step()


def evaluate(model, num_test_windows, test_loader, max_rul, device):
    model.to(device)
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    y_test = (y_test.reshape(-1)) * max_rul
    rul_pred, _ = model.forward(x_test)
    rul_pred = rul_pred.reshape(-1)  # (497,1) -- (497,)
    rul_pred = rul_pred.detach()
    rul_pred *= max_rul
    preds_for_each_engine = torch.split(rul_pred, num_test_windows)
    y_test = torch.split(y_test, num_test_windows)
    y_test = torch.tensor([item.sum() / len(item) for item in y_test])
    y_test, index = y_test.sort(descending=True)
    mean_pred_for_each_engine = torch.tensor([item.sum() / len(item) for item in preds_for_each_engine])
    mean_pred_for_each_engine = torch.index_select(mean_pred_for_each_engine, dim=0, index=index)
    mean_pred_for_each_engine = mean_pred_for_each_engine.floor()
    RMSE = mean_squared_error(y_test, mean_pred_for_each_engine, squared=False).item()
    score = compute_s_score(y_test, mean_pred_for_each_engine)
    return RMSE, score
