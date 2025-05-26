import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader ,Dataset, random_split 
from pathlib import Path
from config import get_config , get_weights_path
from Dataset import BilingualDataset,causal_mask
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from model import build_transformer
import numpy

def greedy_decode(model,src,src_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(src,src_mask)
    # initialize the decoder input
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for the target 
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        
        # calculate the output of the decoder
        out = model.decode(decoder_input,encoder_output,src_mask,decoder_mask)
        
        # get the next token
        prob = model.project(out[:,-1])
        # Select the token with the max prob
        _, next_word = torch.max(prob,dim=1)
        
        decoder_input = torch.cat([decoder_input,torch.empty(1,1).type_as(src).fill_(next_word.item()).to(device)],dim=1)
        
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)
         
    
def run_validation(model, val_ds ,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_step,writer,num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    
    # size of the console window
    console_width = 80
    with torch.no_grad():
        for batch in val_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1
            
            model_out = greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)
            
            source_text = batch['src_text'][0]
            target_text = batch['src_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # print to the console
            print_msg('-'*console_width)
            print_msg(f'Source :{source_text}')
            print_msg(f'target :{target_text}')
            print_msg(f'predicted :{model_out_text}')
            
            if count == num_examples:
                break
            
            
            
def get_all_sentences(ds,lang):
    for ex in ds:
        yield ex['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() ## change this to bytelevel tokenisation
        trainer = BpeTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else :
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    
    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])    
    
    # keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(ds_raw)*0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len']) 
    
    max_len_src = 0
    max_len_tgt = 0
    for ex in ds_raw:
        src_ids = tokenizer_src.encode(ex['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(ex['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of source language: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True) 
    val_dataloader = DataLoader(val_ds,batch_size=config['batch_size'],shuffle=True)
    
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt      

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config["seq_len"],config["seq_len"],config["d_model"])
    return model 

def train_model(config):
    #define the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')
    
    Path(config["model_folder"]).mkdir(parents=True,exist_ok=True)
    
    train_data_loader,val_data_loader,tokenizer_src,tokenizer_tgt = get_ds(config)
    model = get_model(config,len(tokenizer_src.get_vocab()),len(tokenizer_tgt.get_vocab())).to(device)
    
    # tensorboard writer
    writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_path(config,config['preload'])
        print(f"preloading model  {model_filename}")
        state  = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        # model.train()
        batch_iterator = tqdm(train_data_loader,desc=f'processing epoch {epoch : 02d}')
        
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device) ## hide padding tokens
            decoder_mask = batch['decoder_mask'].to(device) ## hide padding tokens and future tokens
            
            # run the tensor through the model
            encoder_output = model.encode(encoder_input,encoder_mask) # (batch_size,seq_len,d_model)
            decoder_output = model.decode(decoder_input,encoder_output,encoder_mask,decoder_mask) # (batch_size,seq_len,d_model)
            project_output = model.project(decoder_output)# (batch_size,seq_len,vocab_size)
            
            label = batch['label'].to(device) # (batch_size,seq_len)
            
            # calculate the loss (batch_size*seq_len,vocab_size) and (batch_size*seq_len)
            loss = loss_fn(project_output.view(-1,project_output.size(-1)),label.view(-1))
            batch_iterator.set_postfix({f" loss":f"{loss.item():6.3f}"})
            
            # log the loss
            writer.add_scalar('train/loss',loss.item(),global_step)
            writer.flush()
            
            # backpropagation the loss
            loss.backward()
            
            # update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            
            global_step += 1
            
        run_validation(model,val_data_loader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,lambda msg:batch_iterator.write(msg),global_step,writer)
        # save the model
        model_filename = get_weights_path(config,f'{epoch:02d}')
        
        torch.save({
            'epoch':epoch,
            'global_step':global_step,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict()
        },model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
            
            
        
    
    
    

