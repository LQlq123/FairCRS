import os
import torch
from loguru import logger
import json
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt
from crslab.evaluator.metrics.rec import *
once = False

class KBRDSystem(BaseSystem):
    """This is the system for KBRD model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KBRDSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        self.user_group = {}

        #TG-ReDial
        with open("data/dataset/tgredial/pkuseg/group/active5.json") as f:
            self.user_group["priority"] = json.load(f)
        with open("data/dataset/tgredial/pkuseg/group/inactive5.json") as f:
            self.user_group["unpriority"] = json.load(f)

        #ReDial
        # with open("data/dataset/redial/nltk/group/active5.json") as f:
        #     self.user_group["priority"] = json.load(f)
        # with open("data/dataset/redial/nltk/group/inactive5.json") as f:
        #     self.user_group["unpriority"] = json.load(f)

        self.uid_group = {uid: group for group, ids in self.user_group.items() for uid in ids}

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, lable in zip(rec_ranks, item_label):
            lable = self.item_ids.index(lable)
            self.evaluator.rec_evaluate(rec_rank, lable)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode, **kwargs):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            # rec_loss = rec_loss.sum()
            if mode == 'train':
            # == User-oriented Fairness Strategy == #
                k = 50
                from pytorchltr.loss import LambdaNDCGLoss1
                loss_fn = LambdaNDCGLoss1()
                scores, indices = torch.sort(rec_scores, dim=-1)
                scores, indices = scores[:, :k], indices[:, :k]
                batch_n = torch.tensor([k for i in range(scores.shape[0])], dtype=torch.long, device="cuda")
                # Batch of nDCG loss (Batch, nDCG@k)
                ndcg_loss = loss_fn(scores, indices, batch_n)
                # self.user_group = {"unpriority": [user_id, ...], "priority": [user_id, ...]}
                user_group_indices = {}
                for i, uid in enumerate(batch["user_ids"]):
                    uid = int(uid)
                    user_group_indices.setdefault(self.uid_group.get(uid, "unknown"), []).append(i)

                # Remember to change the user group path above when changing datasets
                uepsilon = 5 * 1E-4
                priority_lambda = 20
                unpriority_lambda = 50
                user_cond = "priority" in user_group_indices and "unpriority" in user_group_indices
                if user_cond:
                    loss_delta_user_group = (
                            priority_lambda * ndcg_loss[user_group_indices["priority"]].mean()
                            - unpriority_lambda * ndcg_loss[user_group_indices["unpriority"]].mean()
                    ).abs()
                    if torch.isinf(loss_delta_user_group):
                        loss_delta_user_group = 0
                    rec_loss = rec_loss.sum() + uepsilon * loss_delta_user_group
                    logger.info("loss delta user group {}", float(loss_delta_user_group))
                    logger.info("rec loss {}", float(rec_loss))
                    logger.info("uepsilon{},p{},u{}", uepsilon, priority_lambda, unpriority_lambda)
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode != 'test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)
                self.conv_evaluate(preds, batch['response'])

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())
        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')

                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']

                ################################### Test!
                logger.info('[Test]')
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='test')
                    torch.cuda.empty_cache()
                self.evaluator.report(mode='test')

                if self.early_stop(metric):
                   break

        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        elif len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                # early stop
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass
