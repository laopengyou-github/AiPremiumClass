# 模型训练与微调
### 一、差分学习率与动态学习率
##### 1、优化器transformers.AdamW带权重参数的自适应，然后动态学习率选择带warmup的transformers.get_linear_scheduler_with_warmup()，参数说明：1、optimizer (torch.optim.Optimizer)必需参数。训练模型时使用的优化器，例如 AdamW。2、num_warmup_steps (int)必需参数。预热阶段的步数，在这个阶段学习率会从 0 线性增加到初始学习率。3、num_training_steps (int)必需参数。训练的总步数，用于确定学习率衰减的进度。4、last_epoch (int, 可选，默认为 -1)恢复训练时的起始 epoch。如果是从头开始训练，使用默认值 -1。
##### 2、差分学习率：就是将预训练模型参数分成几份然后设置不同的学习率来进行学习。model.name_parameters() ...
### 二、自动混合精度
##### 1、pytorch默认用FP32训练，使用fp16之类的进行混合训练，可以提速降低显存使用，训练开始前声明scaler = torch.gradscaler('cuda') 模型放在with torch.autocast():里，然后scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
### 三、pytroch数据并行训练
##### 1、在多GPU上进行训练，模型一样就是把数据一个大batch128分给4个gpu使用每个32batch size,然后每个里面的GPU都会又分几份(mini-batch)然后进行计算梯度已小份来传递梯度就是不断计算并返回平均的梯度：DDP(distributedDataParallel),每个minibatch需要通信所以需要用torch.distributed,还要设置地址和端口号，设置进程时设置通信协议，nccl(NV),具体实现看代码吧，trainer和trainingargument中也可以实现。
