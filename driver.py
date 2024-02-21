import torch
import numpy as np
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

class Driver():

    def __eval_loop(self, model, batches, counts, classes, data, n_nghs):

        tiles = len(batches)
        l = int(counts.sum() * tiles)

        src_l_cut = np.empty(l, dtype=int)
        dst_l_cut = np.empty(l, dtype=int)
        ts_l_cut = np.empty(l, dtype=int)
        src_utype_l_cut = np.empty(l, dtype=int)
        dst_utype_l_cut = np.empty(l, dtype=int)
        etype_l = np.empty(l, dtype=int)
        lbls = np.empty(l)
        s_idx = 0

        for i, batch in enumerate(batches):
            e_idx = s_idx + int(counts[i] * tiles)
            src_l_cut[s_idx: e_idx] = np.repeat(data.src_l[batch], tiles)
            dst_l_cut[s_idx: e_idx] = np.repeat(data.dst_l[batch], tiles)
            ts_l_cut[s_idx: e_idx] = np.repeat(data.ts_l[batch], tiles)
            src_utype_l_cut[s_idx: e_idx] = np.repeat(data.u_type_l[batch],
                                                    tiles)
            dst_utype_l_cut[s_idx: e_idx] = np.repeat(data.v_type_l[batch],
                                                    tiles)
            etype_slice = np.tile(classes, len(batch))
            etype_l[s_idx: e_idx] = etype_slice
            lbls[s_idx: e_idx] = (etype_slice == classes[i])
            s_idx = e_idx
        prob = model.link_contrast(src_l_cut, dst_l_cut, ts_l_cut,
                                    src_utype_l_cut, dst_utype_l_cut,
                                    etype_l, lbls, n_nghs)
        prob = prob.reshape((len(prob) // tiles, tiles))
        lbls = lbls.reshape((len(lbls) // tiles, tiles))
        prob = prob / prob.sum(1, keepdim=True)
        return prob, lbls, tiles


    def eval(self, model, optimizer, criterion, batch_sampler, data,
             n_nghs, beta, mode):

        start_time = time.time()
        ap, auc, m_loss = [], [], []
        memory_backup = None
        batch_sampler.reset()

        if mode == 'train':
            model = model.train()
            optimizer.zero_grad()
            while True:

                batches, counts, classes = batch_sampler.get_batch_index()
                if counts is None or counts.sum()==0:
                    break
                prob, lbls, tiles = self.__eval_loop(model, batches, counts,
                                                     classes, data, n_nghs)
                
                with torch.no_grad():
                    lbl = torch.from_numpy(lbls).type(torch.float).to(model.device)

                loss = criterion(prob, lbl)
                loss += beta * model.affinity_score.reg_loss()

                loss.bachward()
                optimizer.step()
                with torch.no_grad():
                    model = model.eval()
                    prob = prob.reshape(len(prob) * tiles)
                    lbls = lbls.reshape(len(lbls) * tiles)
                    _ap, _auc = self.evaluate_score(lbls, prob)
                    ap.append(_ap)
                    auc.append(_auc)
                    m_loss.append(loss.item())

                model.memory.detach_memory()
            memory_backup = model.memory.backup_memory()
        if mode == 'test':
            with torch.no_grad():
                model = model.eval()
                while True:
                    prob, lbls, tiles = self.__eval_loop(model, batches, counts,
                                                         classes, data, n_nghs)
                    _ap, _auc = self.evaluate_score(lbls, prob)
                    ap.append(_ap)
                    auc.append(_auc)
        end_time = time.time()
        return ap, auc, m_loss, memory_backup, end_time - start_time
            

    # def optimize(self):


    def evaluate_score(self, labels, prob):
        pred_score = np.array((prob).cpu().detach().numpy())

        auc = roc_auc_score(labels, pred_score)
        ap = average_precision_score(labels, pred_score)
        return ap, auc
