import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
import os
import random
import wandb
import pprint as pp
import torchmetrics
from tqdm import tqdm
from torchvision.models import vgg19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key="1f46f693b026b39ca87896a587a55919e170c98f")
sweep_config = {
    'method': 'grid', 
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            #'values':['resnet50', 'vgg19']
            'value':'resnet50'
        },
        'quant_method': {
            'value': 'naive'
        },
        'load_model': {
            'value': False
        },
        'epochs': { #resnet50:30  vgg19:10
            'value': 10
        },
        'bits_config_dict': {
            # 'value': [30,30,30,30,30,30,30,30]
            'value':
                {"conv2d_w":[30,30,30,30],
                 "conv2d_b":[30,30,30,30],
                 "linear_w":[30],
                 "linear_b":[30]
                }
        }, 
    }
}
pp.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="VGG19-Quant-8-weightbias-partition")

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def init_model(model_name):
    if model_name == 'resnet50':
        model = resnet50(weights=None)
        model.fc = nn.Linear(2048, 10)  
    elif model_name == 'vgg19':
        model = vgg19(weights=None)
        # 修改最后一层以适应 num_classes
        model.classifier[-1] = nn.Linear(4096, 10)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model


def load_model_optimizer(model, optimizer, model_path):
    '''
        加载权重和梯度
    '''
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model and gradients loaded from {model_path}")

    return model, optimizer

def data_init():
    # 初始化数据预处理流程
    trainTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    testTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # 加载训练数据和测试数据
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trainTransforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testTransforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)        
    return train_loader, test_loader

def quantization(grad, grad_max_val, grad_min_val, bit_num, method='naive'):
    if bit_num > 32 or bit_num < 1:
        raise ValueError(f"bit number is greater than 32 or less than 1")
    else:
        n = 2 ** bit_num
        segment_len = (grad_max_val-grad_min_val) / (n-1)
        if segment_len == 0:
            return grad
        
    if method=='naive':
        assert isinstance(grad_min_val, float)
        assert isinstance(grad_max_val, float)
        assert isinstance(bit_num, int)
        
        seg_idxs = ((grad-grad_min_val) // segment_len).to(device)
        assert not torch.isnan(seg_idxs).any().item(), print(f"div 0 segment_len : {segment_len}")
        lower_grad = grad_min_val + seg_idxs*segment_len
        upper_grad = grad_min_val + (seg_idxs+1)*segment_len
        # print(seg_idxs)
        # print(seg_idxs.dtype)

        # print(lower_bound_list)
        # print(upper_bound_list)
        # print(lower_grad)
        # print(upper_grad)
        prob = (upper_grad-grad) / (upper_grad-lower_grad)
        # print(prob)
        random_tensor = torch.rand(grad.size()).to(device)
        assert isinstance(random_tensor,torch.Tensor)
        assert isinstance(prob, torch.Tensor)
        random_tensor = (random_tensor < prob).float()
        # print(random_tensor)
        qgrad = lower_grad * random_tensor + upper_grad * (1.0-random_tensor)
        return qgrad
    
    else:
        raise ValueError

def quant_grad(param_dict:dict, bits_config:list, quant_method:str):
    '''
    具体实现了梯度量化的操作
    1. 将整个模型的参数梯度切分为8份，根据每份参数的平均绝对值对该份参数进行重要性分级
    2. 按照bits_config对每份参数根据重要性进行量化
    '''
    all_grads = []
    grad_shapes = {}
    for name, param in param_dict.items():
        if param.grad is not None:
            all_grads.append(param.grad.flatten())
            grad_shapes[name] = param.grad.shape
    # 汇总梯度
    all_grads = torch.cat(all_grads)    
    grad_numel = all_grads.numel()
    
    splited_grads = []
    splited_grads_mean = []
    splited_grads_min = []
    splited_grads_max = []
    quantized_grads = []
    assert len(bits_config) > 0
    for i in range(len(bits_config)):
        lower_bound = int(1.0/len(bits_config)*i*grad_numel)
        upper_bound = int(1.0/len(bits_config)*(i+1)*grad_numel)
        grads_indices = all_grads[lower_bound:upper_bound]
        splited_grads.append(grads_indices)
        splited_grads_mean.append(torch.sum(torch.abs(grads_indices)).item() / grads_indices.numel())
        splited_grads_min.append(grads_indices.min().item())
        splited_grads_max.append(grads_indices.max().item())
    # 将列表转换为numpy数组
    array1 = np.array(splited_grads_mean)

    # 计算梯度的平均绝对值排名，梯度的平均绝对值最高的排名为0，以此类推
    indices = np.argsort(-array1)
    ranks = np.empty_like(array1, dtype=int)
    ranks[indices] = np.arange(0, len(indices))
    rank_list = ranks.tolist()
    # print("=========================")
    # print("dict info")
    # for name in param_dict.keys():
    #     print(name)
    # print("splited grad info")
    # print("grad num",[g.numel() for g in splited_grads])
    # print("grad max",splited_grads_max)
    # print("grad min",splited_grads_min)
    # print("rank",rank_list)
    # print("bits config",bits_config)

    for i in range(len(bits_config)):
        quant_bit = bits_config[rank_list[i]]
        local_min = splited_grads_min[i]
        local_max = splited_grads_max[i]
        quantized_indices = quantization(splited_grads[i],local_max,local_min,quant_bit,quant_method)
        quantized_grads.append(quantized_indices)
    all_grads = torch.cat(quantized_grads)
    # 将量化后的梯度分配回原来的参数
    start_idx = 0
    for name, param in param_dict.items():
        if param.grad is not None:
            end_idx = start_idx + grad_shapes[name].numel()
            param.grad.copy_(all_grads[start_idx:end_idx].view(grad_shapes[name]))
            start_idx = end_idx

def quant_vgg19(model:nn.Module, bits_config_dict:dict, quant_method:str):
    '''
    features.0.weight
    features.0.bias
    ......
    classifier.6.weight
    classifier.6.bias
    '''
    conv2d_w_param_dicts = {}
    conv2d_b_param_dicts = {}
    linear_w_param_dicts = {}
    linear_b_param_dicts = {}

    for name, param in model.named_parameters():
        if name.startswith("features") and name.endswith("weight"):
            conv2d_w_param_dicts[name] = param
        elif name.startswith("features") and name.endswith("bias"):
            conv2d_b_param_dicts[name] = param
        elif name.startswith("classifier") and name.endswith("weight"):
            linear_w_param_dicts[name] = param
        elif name.startswith("classifier") and name.endswith("bias"):
            linear_b_param_dicts[name] = param
        else:
            raise KeyError
    quant_grad(conv2d_w_param_dicts,bits_config_dict["conv2d_w"],quant_method)
    quant_grad(conv2d_b_param_dicts,bits_config_dict["conv2d_b"],quant_method)
    quant_grad(linear_w_param_dicts,bits_config_dict["linear_w"],quant_method)
    quant_grad(linear_b_param_dicts,bits_config_dict["linear_b"],quant_method)
    return

def quant_resnet(model:nn.Module, bits_config_dict:dict, quant_method:str):
    
    return

def print_grads(model:nn.Module,info:str):
    print("====================================")
    print(info)
    for name, param in model.named_parameters():
        print(name)
        print(param.grad)
    print("====================================")
    return 1

def train_epoch(model, loader, optimizer, criterion, bits_config_dict, quant_method):
    cumu_loss = 0
    accuracy_metric = torchmetrics.Accuracy(task='multiclass',num_classes=10, top_k=1).to(device)    
    
    for iter, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        output = model(data)
        loss = criterion(output, target)
        assert not torch.isnan(loss).item(), print(f"ITER {iter} loss has nan")
        cumu_loss += loss.item()

        accuracy_metric.update(output, target)
        assert isinstance(model, nn.Module)
        # ⬅ Backward pass + weight update
        loss.backward()
        # print_grads(model,f"iter {iter} before")
        for name,param in model.named_parameters():
            assert not torch.isnan(param.grad).any().item(), print(f"ITER {iter} before quant grad of param {name} is NAN | param.grad : {param.grad}")
        print("************* QUANT ****************")
        quant_vgg19(model,bits_config_dict,quant_method) 
        print("************* QUANT ****************")  
        for name,param in model.named_parameters():
            assert not torch.isnan(param.grad).any().item(), print(f"ITER {iter} after quant grad of param {name} is NAN | param.grad : {param.grad}")
        # print_grads(model,f"iter {iter} after")
        optimizer.step()
        for name,param in model.named_parameters():
            assert not torch.isnan(param).any().item(), print(f"ITER {iter} after optim.step() param {name} is NAN")


    train_loss = cumu_loss / len(loader)
    train_accuracy = accuracy_metric.compute().item()
    
    
    return train_loss, train_accuracy

def test_epoch(model, loader, criterion):
    model.eval() 
    cumu_loss = 0
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10, top_k=1).to(device)

    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            # ➡ Forward pass
            output = model(data)
            loss = criterion(output, target)
            cumu_loss += loss.item()

            accuracy_metric.update(output, target)


    test_loss = cumu_loss / len(loader)
    test_accuracy = accuracy_metric.compute().item()
    
    return test_loss, test_accuracy


def train(config=None):
    seed_torch(0)
    with wandb.init(config=config):
        
        config = wandb.config
        
        train_loader, test_loader = data_init()
        
        model = init_model(config.model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        if config.load_model == True:
            model, optimizer = load_model_optimizer(model, optimizer)    
            
        model.to(device)
            
        for epoch in tqdm(range(config.epochs)):
            train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, config.bits_config_dict, config.quant_method)
            test_loss, test_accuracy = test_epoch(model, test_loader, criterion)
            wandb.log({"train loss": train_loss, "train accuracy": train_accuracy, "epoch": epoch})
            wandb.log({"test loss": test_loss, "test accuracy": test_accuracy, "epoch": epoch})

if __name__ == "__main__":
    
    wandb.agent(sweep_id, train, count=6)
            
