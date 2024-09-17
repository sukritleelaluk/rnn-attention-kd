import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, ConcatDataset, Dataset, Subset
from torch.utils.data.dataset import random_split
from torcheval.metrics.functional import binary_auroc
from sklearn.metrics import roc_auc_score

##########
# Device #
##########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)

#################
# Teacher Model #
#################

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, encoder_outputs):
        encoder_transformed = self.W(encoder_outputs)
        score = torch.bmm(encoder_transformed, hidden.unsqueeze(2)).squeeze(-1)
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context_vector, attention_weights


class TeacherGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TeacherGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          bias=True, 
                          batch_first=True, 
                          dropout=0.0, 
                          bidirectional=False)

        self.attention = AttentionModule(hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        encoder_outputs, hidden = self.gru(x)
        last_hidden_state = hidden[-1]
        context_vector, attention_weights = self.attention(last_hidden_state, encoder_outputs)
        output = self.linear(context_vector)
        
        return output, hidden, context_vector, attention_weights
    

class TeacherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TeacherLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bias=True, 
                            batch_first=True, 
                            dropout=0.0, 
                            bidirectional=False)

        self.attention = AttentionModule(hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        encoder_outputs, hidden = self.lstm(x)
        last_hidden_state = hidden[-1]
        last_hidden_state = last_hidden_state.squeeze(0)
        context_vector, attention_weights = self.attention(last_hidden_state, encoder_outputs)
        output = self.linear(context_vector)
        
        return output, hidden, context_vector, attention_weights


def train_teacher(model, train_loader, test_loader, epochs, device, train_graph_path, learning_rate, weight_decay, scheduler_funnction=True, print_results=False):
    criterion_multi = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) 
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    model.train()

    all_losses = []
    all_test_losses = []
    all_lrs = []
    
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        for src, trg in train_loader:
            optimizer.zero_grad()
            
            if use_cuda:
                model = model.cuda()
                src = src.cuda()
                trg = trg.cuda()
            preds, hidden, context_vector, attention_weight = model(src)

            if preds.dim() == 1:
                preds = preds.unsqueeze(1)
                preds = preds.view(1, 2)
            if trg.dim() == 1:
                trg = trg.unsqueeze(1)
            trg = trg.to(torch.float32)

            loss = 0
            loss = criterion_multi(preds, trg)

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}! Let's stop dude!!")
                print(f"Epoch: {epoch}, Loss: {loss}, Source: {src}, Target: {trg}")
                break

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_epoch_loss = epoch_loss / len(train_loader)
        all_losses.append(mean_epoch_loss)

        model.eval()
        with torch.no_grad():
            test_loss = 0
            with torch.no_grad():
                for src_test, trg_test in test_loader:
                    src_test, trg_test = src_test.to(device), trg_test.to(device)
                    preds_test, _, _, _ = model(src_test)
                    
                    if preds_test.dim() == 1:
                        preds_test = preds_test.unsqueeze(1).view(1, 2)
                    if trg_test.dim() == 1:
                        trg_test = trg_test.unsqueeze(1).to(torch.float32)
                    
                    test_loss_batch = criterion_multi(preds, trg)
                    test_loss += test_loss_batch.item()
            
            mean_test_loss = test_loss / len(test_loader)
            all_test_losses.append(mean_test_loss)

        model.train()

        if scheduler_funnction:
            scheduler.step()

        if torch.isnan(loss):
            break

        current_lr = optimizer.param_groups[0]['lr']
        all_lrs.append(current_lr)
        
        if print_results:
            print(f"Epoch: {epoch}, Training Loss: {mean_epoch_loss:.4f}, Test Loss: {mean_test_loss:.4f}")
    
        if train_graph_path is not None:
            train_graph = pd.DataFrame({'train_loss': all_losses, 'test_loss': all_test_losses, 'lr': all_lrs})
            train_graph.to_csv(f'{train_graph_path}.csv', index=False, encoding='utf_8_sig')


#################
# Student Model #
#################


class StudentGRU(nn.Module):
    def __init__(self, input_size, hidden_teacher_size, hidden_student_size, num_layers, num_classes):
        super(StudentGRU, self).__init__()
        self.input_size = input_size
        self.hidden_student_size  = hidden_student_size
        self.hidden_teacher_size  = hidden_teacher_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_student_size, 
                          num_layers=num_layers, 
                          bias=True, 
                          batch_first=True, 
                          dropout=0.0, 
                          bidirectional=False)

        self.attention = AttentionModule(hidden_student_size)

        self.intermediate_hidden = nn.Linear(hidden_student_size, hidden_teacher_size)
        self.intermediate_context = nn.Linear(hidden_student_size, hidden_teacher_size)
        self.linear = nn.Linear(hidden_student_size, num_classes)
    
    def forward(self, x):
        encoder_outputs, hidden_s = self.gru(x)
        last_hidden_state = hidden_s[-1]
        context_vector_s, attention_weights = self.attention(last_hidden_state, encoder_outputs)
        output = self.linear(context_vector_s)

        if self.hidden_teacher_size != self.hidden_student_size:
            context_vector_t = self.intermediate_context(context_vector_s)
        else:
            context_vector_t = context_vector_s
        return output, hidden_s, context_vector_t, context_vector_s, attention_weights


class StudentLSTM(nn.Module):
    def __init__(self, input_size, hidden_teacher_size, hidden_student_size, num_layers, num_classes):
        super(StudentLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_student_size  = hidden_student_size
        self.hidden_teacher_size  = hidden_teacher_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_student_size, 
                            num_layers=num_layers, 
                            bias=True, 
                            batch_first=True, 
                            dropout=0.0, 
                            bidirectional=False)

        self.attention = AttentionModule(hidden_student_size)
        self.intermediate_hidden = nn.Linear(hidden_student_size, hidden_teacher_size)
        self.intermediate_context = nn.Linear(hidden_student_size, hidden_teacher_size)
        self.linear = nn.Linear(hidden_student_size, num_classes)
    
    def forward(self, x):
        encoder_outputs, hidden_s = self.lstm(x)
        last_hidden_state = hidden_s[-1]
        last_hidden_state = last_hidden_state.squeeze(0)
        context_vector_s, attention_weights = self.attention(last_hidden_state, encoder_outputs)
        output = self.linear(context_vector_s)

        if self.hidden_teacher_size != self.hidden_student_size:
            context_vector_t = self.intermediate_context(context_vector_s)
        else:
            context_vector_t = context_vector_s
        return output, hidden_s, context_vector_t, context_vector_s, attention_weights


def train_student(model_t, model_s, train_loader, epochs, device, train_graph_path, learning_rate, weight_decay, alpha=0.0, scheduler_funnction=False, print_results=True):
    criterion_mse = nn.MSELoss(reduction='mean')
    criterion_multi = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_s.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) 
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    model_t.eval()
    model_s.train()

    all_losses = []
    all_hidden = []
    all_context = []
    all_test_losses = []
    all_lrs = []
    
    for epoch in range(1, epochs+1):
        epoch_loss = 0
        hidden_epoch_loss = 0
        context_epoch_loss = 0
        for (inputs_teacher, inputs_student, trg_teacher, trg_student) in train_loader:
            optimizer.zero_grad()
            
            if use_cuda:
                model_t = model_t.cuda()
                model_s = model_s.cuda()
                inputs_teacher = inputs_teacher.cuda()
                inputs_student = inputs_student.cuda()
                trg_teacher = trg_teacher.cuda()
                trg_student = trg_student.cuda()
            
            with torch.no_grad():
                preds_teacher, hidden_teacher, context_teacher, _  = model_t(inputs_teacher) 
            preds_student, hidden_student, context_student, _, _ = model_s(inputs_student)
            
            if inputs_teacher.dim() == 1:
                inputs_teacher = inputs_teacher.unsqueeze(1)
                inputs_teacher = inputs_teacher.view(1, 2)
            if trg_teacher.dim() == 1:
                trg_teacher = trg_teacher.unsqueeze(1)
            trg_teacher = trg_teacher.to(torch.float32)

            if inputs_student.dim() == 1:
                inputs_student = inputs_student.unsqueeze(1)
                inputs_student = inputs_student.view(1, 2)
            if trg_student.dim() == 1:
                trg_student = trg_student.unsqueeze(1)
            trg_student = trg_student.to(torch.float32)
            
            hidden_loss = criterion_mse(hidden_student, hidden_teacher)
            hidden_loss.backward(retain_graph=True)
            hidden_epoch_loss += hidden_loss.item()

            context_loss = criterion_mse(context_student, context_teacher)
            context_loss.backward(retain_graph=True)
            context_epoch_loss += context_loss.item()

            softmax_function = torch.nn.Softmax()
            soft_loss = criterion_multi(preds_student, softmax_function(preds_teacher))

            hard_loss = criterion_multi(preds_student, trg_student)
            loss = hard_loss + (alpha * soft_loss) 

            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}! Let's stop dude!!")
                print(f"Epoch: {epoch}, Loss: {loss}, Source: {inputs_student}, Target: {trg_student}")
                break

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_epoch_hidden = hidden_epoch_loss / len(train_loader)
        mean_epoch_context = context_epoch_loss /  len(train_loader)
        mean_epoch_loss = epoch_loss / len(train_loader)
        all_losses.append(mean_epoch_loss)
        all_hidden.append(mean_epoch_hidden)
        all_context.append(mean_epoch_context)

        if scheduler_funnction:
            scheduler.step()

        if torch.isnan(loss):
            break

        current_lr = optimizer.param_groups[0]['lr']
        all_lrs.append(current_lr)
        
        if print_results:
            print(f"Epoch: {epoch}, KD Loss: {mean_epoch_loss:.4f}, HT Loss: {mean_epoch_hidden:.4f}, CT Loss: {mean_epoch_context:.4f}")
    
        if train_graph_path is not None:
            train_graph = pd.DataFrame({'kd_loss': all_losses, 'ht_loss': all_hidden, 'lr': all_lrs})
            train_graph.to_csv(f'{train_graph_path}.csv', index=False, encoding='utf_8_sig')


##############
# Evaluation #
##############

def model_evaluation_binary_accuracy(outputs, labels, threshold=0.5, confusion_matrix=False):
    softmax_function = torch.nn.Softmax()
    outputs_ar = softmax_function(outputs)
    final_outputs = outputs_ar[:, 1]
    # Get the predicted classes and the actual classes
    final_outputs, labels = final_outputs.cpu(), labels.cpu()
    final_outputs = final_outputs.detach().numpy()
    final_outputs = final_outputs.reshape(-1)

    labels = labels.detach().numpy()
    labels = labels.reshape(-1)

    preds = list()
    
    for i in range(final_outputs.shape[0]):
        if final_outputs[i] >= threshold:
            result = 1
            preds.append(result)
        else:
            result = 0
            preds.append(result)

    preds = np.array(preds)
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for p, c in zip(preds, labels):
        if p == 1 and c == 1:
            TP += 1
        elif p == 0 and c == 0:
            TN += 1
        elif p == 1 and c == 0:
            FP += 1
        elif p == 0 and c == 1:
            FN += 1

    # Calculate Precision, Recall, and F1-score with checks to prevent division by zero
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FN + TN + FP) if (TP + FN + TN + FP) > 0 else 0

    prediction_values = pd.DataFrame({'labels': labels, 'prob_out': final_outputs})

    if confusion_matrix:
        return precision, recall, f1_score, accuracy, TP, FP, TN, FN, prediction_values
    else:
        return precision, recall, f1_score, accuracy, prediction_values
    
    
def model_evaluation_binary_auroc(inputs, target):
    softmax_function = torch.nn.Softmax()
    outputs_ar = softmax_function(inputs)
    outputs = outputs_ar[:, 1]
    return binary_auroc(outputs, target)


def model_evaluation_teacher(model, test_loader, device, threshold=0.5, confusion_matrix=False, print_results=False):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Convert one-hot labels to single integer labels if necessary
            if labels.ndimension() > 1:  # Check if labels are one-hot encoded
                labels = labels.argmax(dim=1).to(device)
            else:
                labels = labels.to(device)

            outputs, _, _, _ = model(inputs)

            all_preds.append(outputs)
            all_labels.append(labels)

    # Concatenate all the predictions and labels
    all_preds = [pred if pred.dim() > 1 else pred.unsqueeze(1) for pred in all_preds]
    all_preds = [pred if pred.size(1) == 2 else pred.t() for pred in all_preds]
    all_preds = torch.cat(all_preds)
    if all_preds.dim() != 1:
        all_preds = all_preds.squeeze(1)
    all_labels = torch.cat(all_labels)

    # Calculate metrics
    precision, recall, f1_score, accuracy, TP, FP, TN, FN, prediction_values = model_evaluation_binary_accuracy(outputs=all_preds, 
                                                                                                                labels=all_labels, 
                                                                                                                threshold=threshold, 
                                                                                                                confusion_matrix=True)
        
    auc_score_torch = model_evaluation_binary_auroc(inputs=all_preds, 
                                                    target=all_labels)
    auc_score = auc_score_torch.to('cpu').detach().numpy().copy()
    
    # Calculate accuracy
    if print_results:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F-measure: {f1_score:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")

    if confusion_matrix:
        return precision, recall, f1_score, accuracy, auc_score, TP, FP, TN, FN, prediction_values
    else:
        return precision, recall, f1_score, accuracy, auc_score, prediction_values


def model_evaluation_student(model, test_loader, device, threshold=0.5, confusion_matrix=False, print_results=False):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            if labels.ndimension() > 1:
                labels = labels.argmax(dim=1).to(device)
            else:
                labels = labels.to(device)

            outputs, _, _, _, _ = model(inputs)
            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = [pred if pred.dim() > 1 else pred.unsqueeze(1) for pred in all_preds]
    all_preds = [pred if pred.size(1) == 2 else pred.t() for pred in all_preds]
    all_preds = torch.cat(all_preds)
    if all_preds.dim() != 1:
        all_preds = all_preds.squeeze(1)
    all_labels = torch.cat(all_labels)

    precision, recall, f1_score, accuracy, TP, FP, TN, FN, prediction_values = model_evaluation_binary_accuracy(outputs=all_preds, 
                                                                                                                labels=all_labels, 
                                                                                                                threshold=threshold, 
                                                                                                                confusion_matrix=True)
        
    auc_score_torch = model_evaluation_binary_auroc(inputs=all_preds, 
                                                    target=all_labels)
    auc_score = auc_score_torch.to('cpu').detach().numpy().copy()
    
    if print_results:
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F-measure: {f1_score:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")

    if confusion_matrix:
        return precision, recall, f1_score, accuracy, auc_score, TP, FP, TN, FN, prediction_values
    else:
        return precision, recall, f1_score, accuracy, auc_score, prediction_values
