import numpy as np
import pandas as pd
import os
import pdb
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, confusion_matrix
import sys
from tqdm import tqdm
from sklearn.metrics import auc
import argparse

fprs = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5]
groups = ['male_male','female_female']
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
emb_map = {}
xvec_map = {}

def compute_scores(df_, eer_threshold_overall=0, agnostic_FLAG=False, emb_FLAG=True):
    if emb_FLAG:
        emb_mapping = emb_map
    else:
        emb_mapping = xvec_map
    similarity_scores= []
    labels = []
    for idx, row in tqdm(enumerate(df_.iterrows())):
        enrol = row[1]['audio_1']
        test = row[1]['audio_2']
        label = row[1]['label']
        if not enrol in emb_mapping.keys():
            print(enrol)
        if not test in emb_mapping.keys():
            print(test)

        sim = 1 - cosine(emb_mapping[enrol],emb_mapping[test])

        similarity_scores.append(sim)
        labels.append(label)
    fpr, tpr, threshold = roc_curve(labels, similarity_scores)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = np.mean((eer1,eer2))

    sim = np.array(similarity_scores)
    labels = np.array(labels)
    if not agnostic_FLAG:
       fpr, fnr = compute_fpr_fnr(sim, labels, eer_threshold_overall)
       return sim, labels, eer, fpr, fnr
    else:
       return sim, labels, eer, eer_threshold

def compute_fpr_fnr(sim,labels_e1, thresh):

    preds = np.zeros(labels_e1.shape[0])
    preds[sim > thresh] = 1
    tn, fp, fn, tp = confusion_matrix(labels_e1, preds).ravel()
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    return fpr, fnr

def compute_fdr(fprs, fnrs, alpha=0.5):
    A = np.absolute(fprs[0]-fprs[1])
    B = np.absolute(fnrs[0]-fnrs[1])
    
    return 1 - (alpha*A + (1-alpha)*B)

def compute_auFDR(fpr_ov, tpr_ov, threshold_ov, sim_g0, sim_g1, labels_g0, labels_g1, 
                  score_dir, emb_FLAG=True, alpha=0.5):
    # FDRs at various thersholds
    fdrs = []
    fnrs = []
    for fpr in tqdm(fprs):
        thresh = threshold_ov[np.nanargmin(np.absolute((fpr_ov-fpr)))]
        fnr = 1 - tpr_ov[np.nanargmin(np.absolute((fpr_ov-fpr)))]
        fpr_g0, fnr_g0 = compute_fpr_fnr(sim_g0, labels_g0, thresh)
        fpr_g1, fnr_g1 = compute_fpr_fnr(sim_g1, labels_g1, thresh)
        fdr = compute_fdr((fpr_g0, fpr_g1), (fnr_g0, fnr_g1), float(alpha))
        fdrs.append(np.round(fdr*100,2))
        fnrs.append(np.round(fnr*100,2))
    auFDR = auc([x*100 for x in fprs], fdrs)
    auFDR_10 =  auc([x*100 for x in fprs[0:10]], fdrs[0:10])
    df = pd.DataFrame(zip(fprs,fdrs, fnrs), columns=['fpr','fdr', 'fnr'])
    if emb_FLAG:
        print("Alpha = {} auFDR auFDR_10".format(alpha))
        print("Embeddings: {} {}\n".format(auFDR, auFDR_10))
        df.to_csv(os.path.join(score_dir, 'fdr_at_fpr_gender_alpha_{}.csv'.format(alpha)), index=None)
    else:
        print("Alpha = {} auFDR auFDR_10".format(alpha))
        print("xvectors: {} {}\n".format(auFDR, auFDR_10))
        df.to_csv(os.path.join(score_dir, 'fdr_at_fpr_gender_alpha_{}.csv'.format(alpha)), index=None)
    return auFDR, auFDR_10

def main(args):
    xvec_FLAG = args.eval_xvector

    # Creating necessary trials for gender-specific evaluations   
    trial_dir = args.trials_root
    trials = os.path.join(trial_dir, 'Test-Combined.csv')
    df = pd.read_csv(trials)
    df['label'] = pd.to_numeric(df['label'])

    df_m = df.loc[df["gender_1"]=='male']
    df_f = df.loc[df["gender_1"]=='female']
    df_m_m = df_m.loc[df_m["gender_2"]=='male']
    df_f_f = df_f.loc[df_f["gender_2"]=='female']
    
    if not os.path.exists(os.path.join(trial_dir,'Test-male-all.csv')):
        df_m.to_csv(os.path.join(trial_dir,'Test-male-all.csv'), index=None)
    if not os.path.exists(os.path.join(trial_dir,'Test-female-all.csv')):
        df_f.to_csv(os.path.join(trial_dir,'Test-female-all.csv'), index=None)
    if not os.path.exists(os.path.join(trial_dir,'Test-male-male.csv')):
        df_m_m.to_csv(os.path.join(trial_dir,'Test-male-male.csv'), index=None)
    if not os.path.exists(os.path.join(trial_dir,'Test-female-female.csv')):
        df_f_f.to_csv(os.path.join(trial_dir,'Test-female-female.csv'), index=None)

    # Create directories to save ASV scores
    scores_dir_base = args.scores_root
    if args.xvector_type=='imbalanced':
        scores_dir_xvec = os.path.join(scores_dir_base,'scores_xvec_imbalanced')
    else:
        scores_dir_xvec = os.path.join(scores_dir_base,'scores_xvec')
    scores_dir = os.path.join(scores_dir_base,'scores_exp_{}_epoch_{}'.format(args.exp_id, args.epoch))
    os.makedirs(scores_dir_xvec, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    # Load extracted embeddings and xvectors
    test_utts = np.load(os.path.join(args.data_root,'test_utts.npy'))
    
    pred_dir_base = '/data/rperi/uai_pytorch/predictions_cv_{}/'.format(args.test_split)
    pred_dir = os.path.join(pred_dir_base,'predictions_exp_{}_epoch_{}'.format(args.exp_id,args.epoch), 'combined')
    e1 = np.load(os.path.join(pred_dir,'emb1.npy'))
    for idx, utt in enumerate(test_utts):
        emb_map[utt] = e1[idx,:]
    if xvec_FLAG:
        xvec = np.load(os.path.join(args.data_root,'test_data.npy'))
        for idx, utt in enumerate(test_utts):
            xvec_map[utt] = xvec[idx,:]


    # Gender-agnostic scoring
    print("Computing Gender-agnostic scores")
    if os.path.exists(os.path.join(scores_dir_xvec, 'sim_xvec_overall.npy')) and os.path.exists(os.path.join(scores_dir, 'sim_e1_overall.npy')) and os.path.exists(os.path.join(scores_dir_xvec, 'labels_overall.npy')):
        sim_e1_ov = np.load(os.path.join(scores_dir, 'sim_e1_overall.npy'))
        labels_ov = np.load(os.path.join(scores_dir_xvec, 'labels_overall.npy'))
        fpr, tpr, threshold = roc_curve(labels_ov, sim_e1_ov)
        fnr = 1 - tpr
        eer_threshold_e1_ov = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_e1_ov  = fpr[np.nanargmin(np.absolute((fnr - fpr))) ]

        if xvec_FLAG:
            sim_xvec_ov = np.load(os.path.join(scores_dir_xvec, 'sim_xvec_overall.npy'))
            fpr, tpr, threshold = roc_curve(labels_ov, sim_xvec_ov)
            fnr = 1 - tpr
            eer_threshold_xvec_ov = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
            eer_xvec_ov = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print("Done scoring Gender-agnostic trials")
    else:
        sim_e1_ov, labels_ov, eer_e1_ov, eer_threshold_e1_ov = compute_scores(df, agnostic_FLAG=True)
        np.save(os.path.join(scores_dir, 'sim_e1_overall'), sim_e1_ov)
        np.save(os.path.join(scores_dir_xvec, 'labels_overall'), labels_ov)
        if xvec_FLAG:
            sim_xvec_ov, labels_xvec_ov, eer_xvec_ov, eer_threshold_xvec_ov = compute_scores(df, agnostic_FLAG=True, emb_FLAG=False)
            np.save(os.path.join(scores_dir_xvec, 'sim_xvec_overall'), sim_xvec_ov)
        print("Done scoring Gender-agnostic trials")

    #Gender-specific scoring
    print("Computing Gender-specific scores")
    if (not os.path.exists(os.path.join(scores_dir, 'sim_e1_male_male.npy'))) or (not os.path.exists(os.path.join(scores_dir, 'sim_e1_female_female.npy'))):
        sim_e1_m, labels_e1_m, eer_e1_m, fpr_e1_m, fnr_e1_m = compute_scores(df_m_m, eer_threshold_e1_ov)
        sim_e1_f, labels_e1_f, eer_e1_f, fpr_e1_f, fnr_e1_f = compute_scores(df_f_f, eer_threshold_e1_ov)
        np.save(os.path.join(scores_dir, 'sim_e1_male_male'), sim_e1_m)
        np.save(os.path.join(scores_dir, 'sim_e1_female_female'), sim_e1_f)
        np.save(os.path.join(scores_dir_xvec, 'labels_male_male'), labels_e1_m)
        np.save(os.path.join(scores_dir_xvec, 'labels_female_female'), labels_e1_f)
         
        print("EER_all EER_Male EER_Female")
        print("Embeddings: {} {} {}\n".format(np.round(eer_e1_ov*100,2), np.round(eer_e1_m*100,2), np.round(eer_e1_f*100,2)))
        
        sim_e1_g0 = sim_e1_m
        sim_e1_g1 = sim_e1_f
        labels_g0 = labels_e1_m
        labels_g1 = labels_e1_f
        print("Done scoring Gender-specific trials")
    else:
        sim_e1 = []
        labels = []
        for group in groups:
            sim_e1.append(np.load(os.path.join(scores_dir, 'sim_e1_{}.npy'.format(group))))
            labels.append(np.load(os.path.join(scores_dir_xvec, 'labels_{}.npy'.format(group))))
        sim_e1_g0 = sim_e1[0]
        sim_e1_g1 = sim_e1[1]
        labels_g0 = labels[0]
        labels_g1 = labels[1]
        print("Done scoring Gender-specific trials")
    if xvec_FLAG:
        if (not os.path.exists(os.path.join(scores_dir_xvec, 'sim_xvec_male_male.npy'))) or (not os.path.exists(os.path.join(scores_dir_xvec, 'sim_xvec_female_female.npy'))):
            print("Computing Gender-specific scores for x-vectors")
            sim_xvec_m, labels_xvec_m, eer_xvec_m, fpr_xvec_m, fnr_xvec_m = compute_scores(df_m_m, eer_threshold_xvec_ov, emb_FLAG=False)
            sim_xvec_f, labels_xvec_f, eer_xvec_f, fpr_xvec_f, fnr_xvec_f = compute_scores(df_f_f, eer_threshold_xvec_ov, emb_FLAG=False)
            np.save(os.path.join(scores_dir_xvec, 'sim_xvec_male_male'), sim_xvec_m)
            np.save(os.path.join(scores_dir_xvec, 'sim_xvec_female_female'), sim_xvec_f)
            print("x-vector: {} {} {}\n".format(np.round(eer_xvec_ov*100,2), np.round(eer_xvec_m*100,2),np.round(eer_xvec_f*100,2)))
            print("Done scoring Gender-specific trials for x-vectors")
        else:
            sim_xvec = []
            for group in groups:
                sim_xvec.append(np.load(os.path.join(scores_dir_xvec, 'sim_xvec_{}.npy'.format(group))))
            sim_xvec_g0 = sim_xvec[0]
            sim_xvec_g1 = sim_xvec[1]
            print("Done scoring Gender-specific trials for x-vectors")

    # Compute area under FDR-FPR curve
    fpr_ov, tpr_ov, threshold_ov = roc_curve(labels_ov, sim_e1_ov)
    aus, au10s = [], []
    for alpha in alphas:
        au, au10 = compute_auFDR(fpr_ov, tpr_ov, threshold_ov, sim_e1_g0, sim_e1_g1, labels_g0, labels_g1, scores_dir, emb_FLAG=True, alpha=alpha)
        aus.append(au)
        au10s.append(au10)
    
    df = pd.DataFrame(zip(alphas,aus, au10s), columns=['alpha','au', 'au10'])
    df.to_csv(os.path.join(score_dir, 'au_fdrs.csv'), index=None)
    if xvec_FLAG:
        fpr_ov, tpr_ov, threshold_ov = roc_curve(labels_ov, sim_xvec_ov)
        aus, aus10 = [],[]
        for alpha in alphas:
            compute_auFDR(fpr_ov, tpr_ov, threshold_ov, sim_xvec_g0, sim_xvec_g1, labels_g0, labels_g1, scores_dir_xvec, emb_FLAG=False, alpha=alpha)
            aus.append(au)
            au10s.append(au10)
        df = pd.DataFrame(zip(alphas,aus, au10s), columns=['alpha','au', 'au10'])
        df.to_csv(os.path.join(score_dir_xvec, 'aufdrs.csv'), index=None)
    pdb.set_trace()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_split', type=str, required=True, help='Whether dev or test')
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--epoch', type=str, required=True)

    parser.add_argument('--trials_root', type=str, required=True,
                        help="Directory containing Test-Combined.csv") # /proj/rperi/UAI/data/trials/CommonVoice/dev

    parser.add_argument('--data_root', type=str, required=True,
                        help="Directory containing test_utts.npy") # /proj/rperi/UAI/data/data_CommonVoice_dev

    parser.add_argument('--scores_root', type=str, required=True,
                        help="Directory to save ASV scores") # /data/rperi/uai_pytorch/scores_CommonVoice_dev
    parser.add_argument('--eval_xvector', default=False, action='store_true')
    parser.add_argument('--xvector_type', type=str, default='balanced',
                        help='Load either xvectors trained on balanced or imbalanced data. Corresponds to the scenarios in https://arxiv.org/abs/2104.14067')

    args = parser.parse_args()
    main(args)

