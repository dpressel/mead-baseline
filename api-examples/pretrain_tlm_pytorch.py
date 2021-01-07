import logging
import time
import os
from argparse import ArgumentParser
import baseline
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, write_json, Average, get_num_gpus_multiworker
from baseline.pytorch.embeddings import *
import baseline.embeddings
from eight_mile.optz import *
from eight_mile.pytorch.layers import save_checkpoint, init_distributed
from eight_mile.pytorch.optz import *
from baseline.pytorch.lm import TransformerLanguageModel, TransformerMaskedLanguageModel
from transformer_utils import MultiFileDatasetReader, on_demand_mlm_masking, get_lr_decay
from eight_mile.pytorch.serialize import load_tlm_npz

logger = logging.getLogger(__file__)


"""Pre-train a Transformer model in PyTorch

The datasets in this program are read in as an `IterableDataset`, typically one line per sample, which
makes it efficient to process even very large datasets that may not fit in core memory.  The datasets are
assumed to be sharded over a set of files for training and validation.

The `preproc-tlm` script can be used upfront to generate pre-processed representations which allows the reader
to simple ingest the sample without any on demand vectorization or masking.  This approach should be preferred
where available.  To run the model in this manner, first run `preproc-tlm`, generating keys `x` and `y` containing
the numeric one-hot values for each token, and then in this script, pass `--preprocessed true`.

If the model is an MLM and the `preprocessed` value is false, on-demand MLM masking is performed.

"""

def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--dataset_key", default="tlm",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--file_type", default='json', help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--ffn_pdrop", type=float, default=0.0, help="Dropout in the dense stack")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="LayerDrop to apply")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_decay_steps", type=int, help="decay steps of lr scheduler")
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of lr scheduler")
    parser.add_argument("--lr_alpha", type=float, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=32, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'epoch', 'ignore'])
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of checkpoints to save per epoch")
    parser.add_argument("--mlm", type=str2bool, default=True, help="Use Masked Language Model (MLM) objective")
    parser.add_argument("--preprocessed", type=str2bool, default=True, help="Has the data already been preprocessed?")
    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument('--rpr_value_on', type=str2bool, default=True,
                        help="In relative attention, whether add positional correction to values in addition to the "
                             "correction to attention matrix")
    parser.add_argument("--windowed_ra", type=str2bool, default=False, help="whether prevent attention beyond rpr_k")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--distributed",
                        type=str2bool,
                        default=False,
                        help="Are we doing distributed training?")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="Local rank for distributed training (-1 means use the environment variables to find)")

    args = parser.parse_args()

    if args.basedir is None:
        args.basedir = 'lm-{}-bpe-{}'.format(args.dataset_key, os.getpid())
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    num_gpus = get_num_gpus_multiworker()
    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    do_on_demand_masking = args.mlm and not args.preprocessed
    if do_on_demand_masking:
        logger.info(f"On-demand masking is turned on")
    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank

    if args.file_type == 'tfrecord':
        reader_type = 'tfrecord'
    elif args.preprocessed:
        reader_type = 'preprocessed'
    else:
        reader_type = 'lang'
    reader = MultiFileDatasetReader(args.nctx, args.subword_model_file, args.subword_vocab_file, f'*.{args.file_type}',
                                    reader_type=reader_type)

    # This looks a bit funny but the streaming reader ignores our vocab and gives us the one from the subword_model
    # However, we do need to get counts from our dataset for validation so we can calculate the perplexity
    vocab = reader.build_vocab([args.valid_file])
    # If we are not using chars, then use 'x' for both input and output
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=args.embed_type)
    vocabs = preproc_data['vocab']

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = {'x': preproc_data['embeddings']}
    logger.info("Loaded embeddings")

    train_set = reader.load(args.train_file, vocabs)
    valid_set = reader.load(args.valid_file, vocabs, distribute=False, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_train_workers)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)

    if args.mlm:
        mask_from = vocabs
        vocab_size = len(mask_from)
        mask_value = mask_from.get("[MASK]")
        if mask_value == -1:
            logger.error("We could not find a suitable masking token in the vocab")
            return

    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        rpr_k = None
    elif len(args.rpr_k) == 1:
        rpr_k = args.rpr_k[0]
    else:
        rpr_k = args.rpr_k

    TLM = TransformerMaskedLanguageModel if args.mlm else TransformerLanguageModel
    model = TLM.create(embeddings,
                       hsz=args.d_model,
                       d_ff=args.d_ff,
                       tie_weights=True,
                       dropout=args.dropout,
                       gpu=False,
                       num_heads=args.num_heads,
                       layers=args.num_layers,
                       rpr_k=rpr_k,
                       d_k=args.d_k,
                       ffn_pdrop=args.ffn_pdrop,
                       windowed_ra=args.windowed_ra,
                       rpr_value_on=args.rpr_value_on,
                       layer_drop=args.layer_drop,
                       src_keys=['x'], tgt_key='x')

    model.to(args.device)
    loss_function = model.create_loss()
    loss_function.to(args.device)

    logger.info("Loaded model and loss")

    steps_per_epoch = len(train_loader) // num_gpus
    valid_steps = len(valid_loader)
    update_on = steps_per_epoch // args.saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch per GPU: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")
    lr_decay = get_lr_decay(args.lr_scheduler, args.lr, steps_per_epoch, args.epochs, logger,
                            decay_steps=args.lr_decay_steps, decay_rate=args.lr_decay_rate, alpha=args.lr_alpha)
    linear_warmup = WarmupLinearSchedulerPyTorch(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRScheduler(linear_warmup, lr_decay, lr=args.lr)

    global_step = 0
    start_epoch = 0
    if args.restart_from:

        if args.restart_from.endswith('npz'):
            load_tlm_npz(model, args.restart_from)
        else:
            model.load_state_dict(torch.load(args.restart_from))
        vec = args.restart_from.split("-")

        if args.restart_tt:
            tick_type = args.restart_tt
        else:
            tick_type = vec[-2]
        step_num = int(vec[-1].split(".")[0])
        if tick_type == 'epoch':
            start_epoch = step_num
            global_step = start_epoch * steps_per_epoch

        elif tick_type == 'step':
            start_epoch = step_num // steps_per_epoch
            global_step = step_num
        else:
            logger.warning(f"The previous tick was {step_num} but command-line specifies to ignore, setting to 0")

        logger.info("Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d, epoch=%d",
                    args.restart_from, global_step, start_epoch+1)

    optimizer = OptimizerManager(model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay)
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.device)
        logger.info("Model located on %s", args.device)

    model_base = os.path.join(args.basedir, 'checkpoint')
    steps = global_step

    for epoch in range(start_epoch, args.epochs):
        avg_loss = Average('average_train_loss')
        metrics = {}
        optimizer.zero_grad()
        start = time.time()
        model.train()
        train_itr = iter(train_loader)
        for i in range(steps_per_epoch):
            batch = next(train_itr)
            steps += 1
            x, y = batch
            inputs = x.to(args.device)
            labels = y.to(args.device)
            if do_on_demand_masking:
                inputs, labels, _ = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
            inputs = {'x': inputs}

            labels = labels.transpose(0, 1).contiguous()
            logits = model(inputs, None)[0].transpose(0, 1).contiguous()
            if args.mlm:
                loss = loss_function(logits, labels)
            else:
                shift_logits = logits[:-1]
                shift_labels = labels[1:]
                loss = loss_function(shift_logits, shift_labels)
            loss.backward()
            avg_loss.update(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % report_on == 0:
                logging.info(avg_loss)

            if (i + 1) % update_on == 0 and args.local_rank < 1:
                elapsed = (time.time() - start)/60
                logging.info('elapsed time this epoch %d min', elapsed)
                logging.info('elapsed step time %f steps/min', i/elapsed)
                logging.info('LR: %f',  optimizer.current_lr)
                save_checkpoint(model, model_base, steps, tick_type='step')

        # How much time elapsed in minutes
        elapsed = (time.time() - start)/60
        train_token_loss = avg_loss.avg
        # This is the average training token-level loss across all machines
        # This is the token-level training perplexity
        train_token_ppl = math.exp(train_token_loss)
        metrics['train_elapsed_min'] = elapsed
        metrics['average_train_loss'] = train_token_loss
        metrics['train_ppl'] = train_token_ppl
        if args.local_rank < 1:
            avg_valid_loss = Average('average_valid_loss')
            start = time.time()
            model.eval()
            valid_itr = iter(valid_loader)
            for j in range(valid_steps):
                batch = next(valid_itr)
                with torch.no_grad():
                    x, y = batch
                    inputs = x.to(args.device)
                    labels = y.to(args.device)

                    if do_on_demand_masking:
                        inputs, labels, _ = on_demand_mlm_masking(inputs, labels, mask_value, vocab_size)
                    inputs = {'x': inputs}
                    labels = labels.transpose(0, 1).contiguous()
                    logits = model(inputs, None)[0].transpose(0, 1).contiguous()
                    if args.mlm:
                        loss = loss_function(logits, labels)
                    else:
                        shift_logits = logits[:-1]
                        shift_labels = labels[1:]
                        loss = loss_function(shift_logits, shift_labels)
                    avg_valid_loss.update(loss.item())

            valid_token_loss = avg_valid_loss.avg
            valid_token_ppl = math.exp(valid_token_loss)

            elapsed = (time.time() - start)/60
            metrics['valid_elapsed_min'] = elapsed
            metrics['average_valid_loss'] = valid_token_loss
            metrics['average_valid_word_ppl'] = valid_token_ppl
            logger.info(metrics)
            save_checkpoint(model, model_base, epoch, save_npz=True)


if __name__ == "__main__":
    train()

