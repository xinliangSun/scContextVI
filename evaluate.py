import scanpy as sc
import numpy as np

from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error


def evaluate_adata(eval_adata, key_dic, topK=100):
    sc.tl.rank_genes_groups(eval_adata, 
                            groupby=key_dic['condition_key'],
                            reference=key_dic['ctrl_key'], 
                            method="wilcoxon")
    
    degs_pred = eval_adata.uns["rank_genes_groups"]["names"][key_dic['pred_key']]
    degs_ctrl = eval_adata.uns["rank_genes_groups"]["names"][key_dic['stim_key']]
    common_degs = list(set(degs_ctrl[0:topK]) & set(degs_pred[0:topK]))
    common_nums = len(common_degs)
    
    top_gene_list = degs_ctrl[:topK]

    ctrl = eval_adata[(eval_adata.obs[key_dic['condition_key']] 
                       == key_dic['ctrl_key'])].to_df()
    
    pred = eval_adata[(eval_adata.obs[key_dic['condition_key']]
                       == key_dic['pred_key'])].to_df()
    
    stim = eval_adata[(eval_adata.obs[key_dic['condition_key']] 
                       == key_dic['stim_key'])].to_df()

    stim_top_degs = stim.loc[:, top_gene_list]
    pred_top_degs = pred.loc[:, top_gene_list]
    
    stim_degs_mean = stim_top_degs.mean().values
    pred_degs_mean = pred_top_degs.mean().values

    # wasserstein_distance
    dist_list_top = []
    for gene in top_gene_list:
        gene_pred = pred.loc[:, gene].values
        gene_case = stim.loc[:, gene].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list_top.append(dist)

    mean_wd = np.mean(dist_list_top)

    # KL
    KL_values = compute_symmetric_kl(stim_top_degs.values, pred_top_degs.values)
    KL_values = round(KL_values, 4)
    # MSE
    mse = mean_squared_error(stim_degs_mean, pred_degs_mean)


    results = {
        "Mse": round(mse, 4),
        "Common DEGs": common_nums, 
        "Wasserstein": round(mean_wd, 4), 
        "KL": KL_values
    }
    
    return results


def compute_symmetric_kl(X, Y, epsilon: float = 1e-8):
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0) + epsilon
    
    mean_y = np.mean(Y, axis=0)
    std_y = np.std(Y, axis=0) + epsilon
    
    kl_xy = np.log(std_y / std_x) + (std_x**2 + (mean_x - mean_y)**2) / (2 * std_y**2) - 0.5
    kl_yx = np.log(std_x / std_y) + (std_y**2 + (mean_y - mean_x)**2) / (2 * std_x**2) - 0.5
    
    symmetric_kl_per_gene = kl_xy + kl_yx
    
    return np.mean(symmetric_kl_per_gene)


def get_wasserstein_distance(eval_adata, case_key='stimulated', pred_key='pred', 
                             top_genes=None, cal_type='sum'):
    """
    '''
    This function is used to calculate the Wasserstein distance between the predicted response and the real response
    :param eval_adata: adata of gene expression containing ctrl,stim, pred
    :param case_key: key of perturbed condition
    :param pred_key: key of predictive condition
    :param top_genes: a list of top DEGs whose Wasserstein distance are to be calculated
    :param cal_type: 'sum' or 'mean'
    :return: the Wasserstein distance between the predicted response and the real response
    '''
    """
    dist_list = []
    dist_list_top = []
    pred = eval_adata[(eval_adata.obs["condition"] == pred_key)].to_df()
    case = eval_adata[(eval_adata.obs["condition"] == case_key)].to_df()

    for i in range(pred.shape[1]):
        gene_pred = pred.iloc[:, i].values
        gene_case = case.iloc[:, i].values
        dist = wasserstein_distance(gene_pred, gene_case)
        dist_list.append(dist)

    if top_genes is None:
        res = None
        if cal_type == 'mean':
            res = np.mean(dist_list)
        elif cal_type == 'sum':
            res = np.sum(dist_list)
        return res
    else:
        for gene in top_genes:
            gene_pred = pred.loc[:, gene].values
            gene_case = case.loc[:, gene].values
            dist = wasserstein_distance(gene_pred, gene_case)
            dist_list_top.append(dist)
        res = None
        if cal_type == 'mean':
            res = [np.mean(dist_list), np.mean(dist_list_top)]
        elif cal_type == 'sum':
            res = [np.sum(dist_list), np.sum(dist_list_top)]
        return res
