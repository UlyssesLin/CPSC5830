import torch
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from model.module import THAN

def evaluate_score(labels, prob):
    pred_score = np.array((prob).cpu().detach().numpy())

    auc = roc_auc_score(labels, pred_score)
    ap = average_precision_score(labels, pred_score)
    return ap, auc

def _eval_loop(model, batches, counts, classes, data, n_nghs):
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
        lbls[s_idx: e_idx] = (etype_slice == classes[i]).astype(np.float64)
        s_idx = e_idx

    prob = model.link_contrast(src_l_cut, dst_l_cut, ts_l_cut,
                                       src_utype_l_cut, dst_utype_l_cut,
                                       etype_l, lbls, n_nghs)

    prob = prob.reshape((len(prob) // tiles, tiles))
    lbls = lbls.reshape((len(lbls) // tiles, tiles))
    prob = prob / prob.sum(1, keepdim=True)
    return prob, lbls, tiles


def test_eval(hint, model: THAN, batch_sampler, device, data, logger, n_nghs):
    if logger:
        logger.info(hint)
    test_ap, test_auc, test_acc = [], [], []
    with torch.no_grad():
        model = model.eval()
        batch_sampler.reset()
        while True:
            batches, counts, classes = batch_sampler.get_batch_index()
            if counts is None or counts.sum()==0:
                break
            prob, lbls, tiles = _eval_loop(model, batches, counts, classes, data, n_nghs)

            lbl = torch.from_numpy(lbls).type(torch.float).to(device)

            corr = lbl[prob == torch.amax(prob, 1, keepdim=True)]
            _acc = corr.sum() / len(corr)
            test_acc.append(_acc.cpu())

            prob = prob.reshape(len(prob) * tiles)
            lbls = lbls.reshape(len(lbls) * tiles)

            ap, auc = evaluate_score(lbls, prob)
            test_ap.append(ap)
            test_auc.append(auc)

    return np.mean(test_auc), np.mean(test_ap), np.mean(test_acc)

def train_eval(model, batch_sampler, optimizer, criterion, beta, device, data, n_nghs):
    ap, auc, acc, m_loss = [], [], [], []
    batch_sampler.reset()
    while True:
            batches, counts, classes = batch_sampler.get_batch_index()
            if counts is None or counts.sum()==0:
                break
            prob, lbls, tiles = _eval_loop(model, batches, counts, classes, data, n_nghs)

            with torch.no_grad():
                lbl = torch.from_numpy(lbls).type(torch.float).to(device)

            optimizer.zero_grad()
            model = model.train()

            loss = criterion(prob, lbl)
            loss += beta * model.affinity_score.reg_loss()

            loss.backward()
            optimizer.step()

            corr = lbl[prob == torch.amax(prob, 1, keepdim=True)]
            _acc = corr.sum() / len(corr)
            acc.append(_acc.cpu())
            with torch.no_grad():
                model = model.eval()
                prob = prob.reshape(len(prob) * tiles)
                lbls = lbls.reshape(len(lbls) * tiles)
                _ap, _auc = evaluate_score(lbls, prob)
                ap.append(_ap)
                auc.append(_auc)
                m_loss.append(loss.item())
            model.memory.detach_memory()
    return auc, ap, acc, m_loss