import os
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataset import CustomDataset
from autoencoder import AutoEncoder, get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer():
    
    def __init__(self, args, model, optimizer, train_loader, val_loader, scheduler, device, epoch, rank):
        
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.epoch = epoch
        self.rank = rank
        
        # Loss Function
        self.criterion = nn.L1Loss().to(self.device)
        self.anomaly_calculator = nn.L1Loss(reduction="none").to(self.device)
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def fit(self, ):
        
        self.model.to(self.device)
        best_score = 0
        
        for epoch in range(self.epoch):
            # DDP에서는 각 epoch마다 sampler 설정 필요 (DistributedSampler만)
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            self.model.train()
            train_loss = []
            
            for time, x, y in self.train_loader:
                time, x = time.to(self.device), x.to(self.device)
                
                self.optimizer.zero_grad()

                # 모델 호출 수정 (중복 호출 제거)
                t_emb, _x = self.model(time, x)
                x = torch.cat([t_emb, x], dim=1)
                
                loss = self.criterion(x, _x)
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            # 로그는 rank 0에서만 출력
            if epoch % 10 == 0 and self.rank == 0:
                score = self.validation(self.model, 0.95)
                diff = self.cos(x, _x).cpu().tolist()
                print(f'Epoch : [{epoch}] Train loss : [{np.mean(train_loss)}], Train cos : [{np.mean(diff)}] Val cos : [{score}])')

            if self.scheduler is not None:
                # 모든 프로세스에서 validation score 계산
                score = self.validation(self.model, 0.95)
                self.scheduler.step(score)

            # 모델 저장은 rank 0에서만 (단일 GPU에서는 항상 저장)
            if (self.rank == 0 or not dist.is_initialized()) and best_score < score:
                best_score = score
                # DDP에서는 .module을 통해 원본 모델에 접근
                if hasattr(self.model, 'module'):
                    torch.save(self.model.module.state_dict(), os.path.join(self.args.model_dir, "best_model.pth"), _use_new_zipfile_serialization=False)
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, "best_model.pth"), _use_new_zipfile_serialization=False)
                
        return self.model
    
    def validation(self, eval_model, thr):
        
        eval_model.eval()
        total_diff = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for time, x, y in self.val_loader:
                time, x, y = time.to(self.device), x.to(self.device), y.to(self.device)
                
                t_emb, _x = self.model(time, x)
                x = torch.cat([t_emb, x], dim=1)
                
                # 배치별로 cosine similarity 계산
                batch_diff = self.cos(x, _x)
                total_diff += batch_diff.sum().item()
                total_samples += batch_diff.size(0)
        
        # 단순히 로컬 결과만 반환 (DDP 통신 제거)
        return total_diff / total_samples if total_samples > 0 else 0.0

def setup_ddp():
    """DDP 초기화"""
    # SageMaker에서 제공하는 환경변수들
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 단일 GPU만 있는 경우 DDP 건너뛰기
    if world_size == 1:
        return local_rank, rank, world_size
    
    # CUDA 디바이스 설정
    torch.cuda.set_device(local_rank)
    
    # Process group 초기화 (NCCL 문제시 gloo 백엔드 사용)
    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)  # 타임아웃 설정
        )
    except Exception as e:
        print(f"NCCL 초기화 실패, gloo 백엔드로 전환: {e}")
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=30)
        )
    
    return local_rank, rank, world_size

def cleanup_ddp():
    """DDP 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

def check_gpu():
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"# DEVICE {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        print("- Memory Usage:")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(local_rank)/1024**3,1)} GB")
        print(f"  Cached:    {round(torch.cuda.memory_reserved(local_rank)/1024**3,1)} GB\n")
        
        device = f"cuda:{local_rank}"
    else:
        print("# GPU is not available")
        device = "cpu"

    print(f'# Current cuda device: {torch.cuda.current_device()}')
    
    return device

def from_pickle(obj_path):

    with open(file=obj_path, mode="rb") as f:
        obj=pickle.load(f)

    return obj

def get_and_define_dataset(args):
    
    train_x_scaled_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_x_scaled_shingle.pkl"
        )
    )
    
    train_y_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_y_shingle.pkl"
        )
    )

    train_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )

    test_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )
    
    return train_ds, test_ds

def get_dataloader(args, train_ds, test_ds, world_size):
    
    # DDP를 위한 DistributedSampler 사용
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=int(os.environ.get("RANK", 0)),
            shuffle=True
        )
        val_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=int(os.environ.get("RANK", 0)),
            shuffle=False
        )
        shuffle = False  # sampler 사용시 shuffle=False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3
    )
    
    return train_loader, val_loader

def train(args):
    
    # DDP 설정
    local_rank, rank, world_size = setup_ddp()
    
    # rank 0에서만 로그 출력
    if rank == 0:
        logger.info("Check gpu..")
    
    device = check_gpu()
    
    if rank == 0:
        logger.info(f"Device Type: {device}")
        logger.info(f"World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")
    
    if rank == 0:
        logger.info("Load and define dataset..")
    train_ds, test_ds = get_and_define_dataset(args)
    
    if rank == 0:
        logger.info("Define dataloader..")
    train_loader, val_loader = get_dataloader(args, train_ds, test_ds, world_size)
    
    if rank == 0:
        logger.info("Set components..")

    # 모델을 먼저 device로 이동 후 DDP로 래핑
    model = get_model(
        input_dim=args.num_features*args.shingle_size + args.emb_size,
        hidden_sizes=[64, 48],
        btl_size=32,
        emb_size=args.emb_size
    ).to(device)
    
    # DDP로 래핑 (단일 GPU인 경우 건너뛰기)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        threshold_mode='abs',
        min_lr=1e-8,
        verbose=True if rank == 0 else False  # rank 0에서만 verbose
    )
    
    if rank == 0:
        logger.info("Define trainer..")
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        epoch=args.epochs,
        rank=rank
    )
    
    if rank == 0:
        logger.info("Start training..")
    
    try:
        model = trainer.fit()
    finally:
        # 정리
        cleanup_ddp()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=int(os.environ.get("SM_HP_WORKERS", 2)), metavar="W", help="number of data loading workers (default: 2)")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SM_HP_EPOCHS", 150)), metavar="E", help="number of total epochs to run (default: 150)")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("SM_HP_BATCH_SIZE", 512)), metavar="BS", help="batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=float(os.environ.get("SM_HP_LR", 0.001)), metavar="LR", help="initial learning rate (default: 0.001)")
    
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--val_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 1)))
    
    parser.add_argument("--shingle_size", type=int, default=int(os.environ.get("SM_HP_SHINGLE_SIZE", 10)))
    parser.add_argument("--num_features", type=int, default=int(os.environ.get("SM_HP_NUM_FEATURES", 10)))
    parser.add_argument("--emb_size", type=int, default=int(os.environ.get("SM_HP_EMB_SIZE", 32)))
        
    train(parser.parse_args())