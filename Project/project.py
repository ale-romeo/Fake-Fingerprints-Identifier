import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize as opt
from scipy.special import logsumexp
import pickle
import os
import pandas as pd

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, -1))

def vcol(v):
    return v.reshape(-1, 1)

def load_projectData():
    DList = []
    labelsList = []

    with open("trainData.txt") as f:
        for line in f:
            try:
                features = line.split(",")[0:-1]
                features = mcol(np.array([float(i) for i in features]))
                class_ = True if int(line.split(",")[-1]) == 1 else False
                DList.append(features)
                labelsList.append(class_)
            except:
                pass
            
    return np.hstack(DList), np.array(labelsList, dtype=bool)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def save_model(classifier, model, filename, model_params, validation_scores):
    directory = f"models/{classifier}/{model}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    data = {'model_params': model_params, 'validation_scores': validation_scores}
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def load_model(classifier, model, filename):
    filepath = f"models/{classifier}/{model}/{filename}"
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def eval_mu(D, param=1):
    return D.mean(param)

def eval_cov(D, param=1):
    mu = eval_mu(D, param)
    DC = D - mcol(mu)
    N = float(D.shape[param])
    C = 1/N*DC@DC.T
    return C

def Sw(D, L, array):
    Sw_sum = 0

    for _ in array:
        D_c = D[:, L==_]
        DC_c = D_c - mcol(D_c.mean(1))
        Sw_sum += D_c.shape[1] * (1/D_c.shape[1]) * DC_c @ DC_c.T

    return Sw_sum/float(D.shape[1])

def PCA(DTR, m, DVAL=None):
    C = eval_cov(DTR)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DVAL_PCA = np.dot(P.T, DVAL) if DVAL is not None else None
    return np.dot(P.T, DTR), DVAL_PCA

def LDA(DTR, L, array, DVAL=None):
    Sw_sum = 0
    Sb_sum = 0

    for _ in array:
        D_c = DTR[:, L==_]
        SW_c = eval_cov(D_c)
        Sw_sum += SW_c*float(D_c.shape[1])
        mu_p = (mcol(D_c.mean(1))-mcol(DTR.mean(1)))
        Sb_sum += float(D_c.shape[1])*mu_p*mu_p.T
    Sw = Sw_sum/float(DTR.shape[1])
    Sb = Sb_sum/float(DTR.shape[1])

    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:1]
    DVAL_LDA = np.dot(W.T, DVAL) if DVAL is not None else None
    return np.dot(W.T, DTR), DVAL_LDA

def centerData(DTR, DVAL):
    mu = eval_mu(DTR)
    DTR = DTR - mcol(mu)
    DVAL = DVAL - mcol(mu)
    return DTR, DVAL

def zNormData(DTR, DVAL):
    mu = eval_mu(DTR)
    sigma = DTR.std(1)
    DTR = (DTR - mcol(mu)) / mcol(sigma)
    DVAL = (DVAL - mcol(mu)) / mcol(sigma)
    return DTR, DVAL

def logS(num_classes, num_samples, DTR, LTR, DTE, version):
    log_S = np.zeros((num_classes, num_samples))
    for cls in [0, 1, 2]:
        D_cls = DTR[:, LTR==cls]
        if version == "naive":
            C = np.diag(np.diag(eval_cov(D_cls)))
        elif version == "tied":
            C = Sw(DTR, LTR, [0,1,2])
        else:
            C = eval_cov(D_cls)
        log_S[cls, :] = logpdf_GAU_ND(DTE, mcol(eval_mu(D_cls)), C)
    logSJoint = logS + mcol(np.log(1/3))
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return log_S, logSJoint, logSMarginal, logSPost

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]
    sign_log_det, log_det = np.linalg.slogdet(C)
    diff = X - mu
    inner_term = np.dot(np.dot(diff.T, np.linalg.inv(C)), diff)
    log_densities = -0.5 * (M * np.log(2 * np.pi) + log_det + inner_term.diagonal())
    
    return log_densities

def logpdf_GAU_1D(X, mu, C):
    M = 1
    sign_log_det, log_det = np.linalg.slogdet(C)
    diff = X - mu
    inner_term = np.dot(np.dot(diff.T, np.linalg.inv(C)), diff)
    log_densities = -0.5 * (M * np.log(2 * np.pi) + log_det + inner_term.diagonal())
    
    return log_densities

def llr_binary(num_classes, num_samples, DTR, LTR, DTE, version):
    logS = np.zeros((num_classes, num_samples))

    for cls, _ in zip([False, True], [0, 1]):
        D_cls = DTR[:, LTR==cls]
        if version == "naive":
            C = np.diag(np.diag(eval_cov(D_cls)))
        elif version == "tied":
            C = Sw(DTR, LTR, [False, True])
        else:
            C = eval_cov(D_cls)
        logS[_, :] = logpdf_GAU_ND(DTE, mcol(eval_mu(D_cls)), C)

    llrs = logS[1] - logS[0]
    predictions = np.where(llrs >= 0, True, False)
    return predictions, llrs

def errorRateLab3(DVAL_lda, DTR_lda, LTR, LVAL, threshold=0):
    if threshold == 0:
        threshold = (DTR_lda[0, LTR==False].mean() + DTR_lda[0, LTR==True].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = True
    PVAL[DVAL_lda[0] < threshold] = False
    
    cnt = 0
    for _ in range(len(LVAL)):
        if LVAL[_] != PVAL[_]:
            cnt += 1
    return cnt/len(LVAL)

def optBayesDecisions(llr, pi1, Cfn, Cfp):
    threshold = -np.log(pi1 * Cfn / ((1 - pi1) * Cfp))
    decisions = np.where(llr > threshold, True, False)
    decisions = np.where(llr > threshold, True, False)
    return decisions

def confMatrix(predictions, labels):
    TP = np.sum((predictions == True) & (labels == True))
    TN = np.sum((predictions == False) & (labels == False))
    FP = np.sum((predictions == True) & (labels == False))
    FN = np.sum((predictions == False) & (labels == True))
    conf_matrix = np.array([[TN, FP], [FN, TP]])
    return conf_matrix

def bayesRisk(pi1, Cfn, Cfp, conf_matrix):
    Pfn = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    Pfp = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
    B = pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp
    return B

def normDCF(pi1, Cfn, Cfp, conf_matrix):
    B_dummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
    B = bayesRisk(pi1, Cfn, Cfp, conf_matrix)
    DCF = B / B_dummy
    return DCF

def minDCF(llr, labels, pi1, Cfn, Cfp):
    thresholds = np.unique(llr)  # Ordina e rimuove i duplicati
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))  # Aggiungi -∞ e +∞
    
    min_dcf = float('inf')
    for threshold in thresholds:
        decisions = np.where(llr > threshold, True, False)
        conf_matrix = confMatrix(decisions, labels)
        dcf = normDCF(pi1, Cfn, Cfp, conf_matrix)
        min_dcf = min(min_dcf, dcf)
    
    return min_dcf

def pieffvsDCFs(llr, labels, eff_prior_log_odds, Cfn=1, Cfp=1):
    pi_eff = 1 / (1 + np.exp(-eff_prior_log_odds))
    actual_dcf_values = []
    min_dcf_values = []

    for pi_eff_value in pi_eff:
        act_dcf = normDCF(pi_eff_value, Cfn, Cfp, confMatrix(optBayesDecisions(llr, pi_eff_value, Cfn, Cfp), labels))
        min_dcf = minDCF(llr, labels, pi_eff_value, Cfn, Cfp)

        actual_dcf_values.append(act_dcf)
        min_dcf_values.append(min_dcf)

    return actual_dcf_values, min_dcf_values

def bestmbyDCF(DTR, LTR, DVAL, LVAL, pi1, Cfn=1, Cfp=1):
    best_DCF = float('inf')
    best_min_DCF = float('inf')

    for m in range(1, DTR.shape[0] + 1):
        DTR_PCA, DVAL_PCA = PCA(DTR, m, DVAL)

        best_DCF_version = float('inf')
        best_min_DCF_version = float('inf')

        for version in ["gaussian", "tied", "naive"]:
            predictions, llrs = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            act_dcf = normDCF(pi1, 1, 1, confMatrix(optBayesDecisions(llrs, pi1, Cfn, Cfp), LVAL))
            min_dcf = minDCF(llrs, LVAL, pi1, Cfn, Cfp)

            if act_dcf < best_DCF_version:
                best_DCF_version = act_dcf
                best_min_DCF_version = min_dcf

        if best_DCF_version < best_DCF:
            best_DCF = best_DCF_version
            best_min_DCF = best_min_DCF_version
            best_m = m

    return best_m, best_DCF, best_min_DCF

def logreg_obj(v, DTR, LTR, l):
    w = v[:-1]
    b = v[-1]
    ZTR = 2 * LTR - 1  # Converte le etichette in {1, -1}
    S = (np.dot(w.T, DTR) + b).ravel()
    
    # Calcolo dell'obiettivo
    loss = np.mean(np.logaddexp(0, -ZTR * S))
    reg_term = (l / 2) * np.sum(w**2)
    J = reg_term + loss
    
    # Calcolo del gradiente
    G = -ZTR / (1.0 + np.exp(ZTR * S))
    grad_w = l * w + np.mean(G * DTR, axis=1)
    grad_b = np.mean(G)
    grad = np.append(grad_w, grad_b)
    
    return J, grad

def logreg_obj_weighted(v, DTR, LTR, l, pi_T):
    w = v[:-1]
    b = v[-1]
    ZTR = 2 * LTR - 1  # Converte le etichette in {1, -1}
    S = (np.dot(w.T, DTR) + b).ravel()
    
    nT = np.sum(LTR == 1)
    nF = np.sum(LTR == 0)
    weights = np.where(ZTR == 1, pi_T / nT, (1 - pi_T) / nF)
    
    # Calcolo dell'obiettivo
    loss = np.sum(weights * np.logaddexp(0, -ZTR * S))
    reg_term = (l / 2) * np.sum(w**2)
    J = reg_term + loss
    
    # Calcolo del gradiente
    G = -ZTR / (1.0 + np.exp(ZTR * S))
    grad_w = l * w + np.sum(weights * G * DTR, axis=1)
    grad_b = np.sum(weights * G)
    grad = np.append(grad_w, grad_b)
    
    return J, grad

def train_logreg(DTR, LTR, l, pi_T = 0, weighted=False):
    x0 = np.zeros(DTR.shape[0] + 1)
    if weighted:
        result = opt.fmin_l_bfgs_b(logreg_obj_weighted, x0, args=(DTR, LTR, l, pi_T), approx_grad=False)
    else:
        result = opt.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l), approx_grad=False)
    return result[0], result[1]

def llrScores(D, w, b, pi_emp, pi1, Cfn, Cfp):
    scores = np.dot(w.T, D) + b
    llr_scores = scores - np.log(pi_emp / (1 - pi_emp))
    predictions = np.where(llr_scores > -np.log(pi1 * Cfn / ((1 - pi1) * Cfp)), True, False)
    return predictions, llr_scores

def LogRegression(DTR, LTR, DVAL, LVAL, lambdas, pi_T, pi_emp, model, Cfn=1, Cfp=1):
    actual_dcf_values = []
    min_dcf_values = []
    combined_scores = []

    weighted = True if model == 'weighted' else False
    if model == 'reduced':
        DTR, LTR = DTR[:, ::50], LTR[::50]
    if model == 'quadratic':
        DTR = expand_features(DTR.T).T
        DVAL = expand_features(DVAL.T).T
    if model == 'centered':
        DTR, DVAL = centerData(DTR, DVAL)
    if model == 'znorm':
        DTR, DVAL = zNormData(DTR, DVAL)
    if model == 'pca':
        DTR, DVAL = PCA(DTR, 5, DVAL)

    for l in lambdas:
        optimal_params, _ = train_logreg(DTR, LTR, l, pi_T, weighted)
        w_opt = optimal_params[:-1]
        b_opt = optimal_params[-1]
        
        predictions, llr_scores = llrScores(DVAL, w_opt, b_opt, pi_emp, pi_T, Cfn, Cfp)
        actual_dcf = normDCF(pi_T, Cfn, Cfp, confMatrix(predictions, LVAL))
        min_dcf = minDCF(llr_scores, LVAL, pi_T, Cfn, Cfp)
        
        actual_dcf_values.append(actual_dcf)
        min_dcf_values.append(min_dcf)

        # Calculate the combined score
        combined_score = 0.3 * actual_dcf + 0.7 * min_dcf
        combined_scores.append(combined_score)
        # Save model parameters and validation scores
        if combined_score == min(combined_scores):
            best_model_params = {'weights': w_opt, 'bias': b_opt}
            best_model_llr_scores = llr_scores
        
    save_model('logreg', model, 'best_model.pkl', best_model_params, best_model_llr_scores)

    return actual_dcf_values, min_dcf_values, combined_scores

def expand_features(X):
    n_samples, n_features = X.shape
    expanded_features = [X]
    expanded_features = [X]
    
    for i in range(n_features):
        expanded_features.append(X[:, i:i+1] ** 2)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            expanded_features.append(X[:, i:i+1] * X[:, j:j+1])
    
    return np.hstack(expanded_features)

def dual_objective(alpha, Hc):
    return 0.5 * np.dot(alpha.T, np.dot(Hc, alpha)) - np.sum(alpha.T)

def dual_gradient(alpha, Hc):
    return np.dot(Hc, alpha) - np.ones_like(alpha)

def predictLinearSVM(DVAL, w, b, K):
    scores = np.dot(w.T, DVAL) + b*K
    return np.where(scores > 0, True, False), scores

def trainLinearSVM(DTR, LTR, Hc, C):
    n = DTR.shape[1]
    bounds = [(0, C) for _ in range(n)]
    alpha_0 = np.zeros(n)
    
    alpha_star, _, _ = opt.fmin_l_bfgs_b(dual_objective, alpha_0, fprime=dual_gradient, bounds=bounds, args=(Hc,), factr=1.0)
    
    w_star = np.sum((alpha_star * LTR) * DTR, axis=1)
    w = w_star[:-1]
    b = w_star[-1]
    return w, b

def sampleScorePolyKSVM(DTR, LTR, alpha_star, x, d, c, xi):
    score = 0
    for i in range(DTR.shape[1]):
        score += alpha_star[i] * LTR[i] * ((np.dot(DTR[:, i].T, x) + c) ** d + xi)
    return score

def predictPolyKSVM(DTR, LTR, alpha_star, DVAL, d, c, xi):
    scores = np.array([sampleScorePolyKSVM(DTR, LTR, alpha_star, DVAL[:, i], d, c, xi) for i in range(DVAL.shape[1])])
    return np.where(scores > 0, True, False), scores

def trainPolyKSVM(DTR, Hc, C):
    n = DTR.shape[1]
    bounds = [(0, C) for _ in range(n)]
    alpha_0 = np.zeros(n)
    
    alpha_star, _, _ = opt.fmin_l_bfgs_b(dual_objective, alpha_0, fprime=dual_gradient, bounds=bounds, args=(Hc,), factr=1.0)
    return alpha_star

def matrixPolyK(D, L, d, c, xi):
    n = D.shape[1]
    Hc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Hc[i, j] = L[i] * L[j] * ((np.dot(D[:, i].T, D[:, j]) + c) ** d + xi)
    return Hc

def sampleScoreRBFKSVM(DTR, LTR, alpha_star, x, gamma, xi):
    score = 0
    for i in range(DTR.shape[1]):
        diff = DTR[:, i] - x
        score += alpha_star[i] * LTR[i] * (np.exp(-gamma * np.dot(diff.T, diff)) + xi)
    return score

def predictRBFKSVM(DTR, LTR, alpha_star, DVAL, gamma, xi):
    scores = np.array([sampleScoreRBFKSVM(DTR, LTR, alpha_star, DVAL[:, i], gamma, xi) for i in range(DVAL.shape[1])])
    return np.where(scores > 0, True, False), scores

def trainRBFKSVM(DTR, Hc, C):
    n = DTR.shape[1]
    bounds = [(0, C) for _ in range(n)]
    alpha_0TR = np.zeros(n)
    
    alpha_starTR, _, _ = opt.fmin_l_bfgs_b(dual_objective, alpha_0TR, fprime=dual_gradient, bounds=bounds, args=(Hc,), factr=1.0)
    return alpha_starTR

def matrixRBFK(D, L, gamma, xi):
    n = D.shape[1]
    Hc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = D[:, i] - D[:, j]
            Hc[i, j] = L[i] * L[j] * (np.exp(-gamma * np.dot(diff.T, diff)) + xi)
    return Hc

def SVM(DTR, LTR, DVAL, LVAL, C_values, kernel_version, params, pi1=0.1, xi=0, K=1.0):
    min_dcf_values = []
    act_dcf_values = []
    combined_scores = []
    d, c, gamma = 0, 0, 0

    LTR = np.where(LTR == True, 1, -1)

    if kernel_version == 'poly':
        d, c = params
        Hc = matrixPolyK(DTR, LTR, d, c, xi)
    if kernel_version == 'rbf':
        gamma = params
        Hc = matrixRBFK(DTR, LTR, gamma, xi)
    if kernel_version == 'centered':
        DTR, DVAL = centerData(DTR, DVAL)
    if kernel_version == 'linear' or kernel_version == 'centered':
        DTR_ext = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
        Hc = np.dot(DTR_ext.T * LTR[:, None], (DTR_ext.T * LTR[:, None]).T)
    
    for C in C_values:
        predictions, scores = [], []
        if kernel_version == 'linear' or kernel_version == 'centered':
            w, b = trainLinearSVM(DTR_ext, LTR, Hc, C)
            predictions, scores = predictLinearSVM(DVAL, w, b, K)
        elif kernel_version == 'poly':
            alpha_star = trainPolyKSVM(DTR, Hc, C)
            predictions, scores = predictPolyKSVM(DTR, LTR, alpha_star, DVAL, d, c, xi)
        elif kernel_version == 'rbf':
            alpha_star = trainRBFKSVM(DTR, Hc, C)
            predictions, scores = predictRBFKSVM(DTR, LTR, alpha_star, DVAL, gamma, xi)

        min_dcf = minDCF(scores, LVAL, pi1, 1, 1)
        act_dcf = normDCF(pi1, 1, 1, confMatrix(optBayesDecisions(scores, pi1, 1, 1), LVAL))
        
        min_dcf_values.append(min_dcf)
        act_dcf_values.append(act_dcf)
        
        # Calculate the combined score
        combined_score = 0.3 * act_dcf + 0.7 * min_dcf
        combined_scores.append(combined_score)
        # Save model parameters and validation scores
        if combined_score == min(combined_scores):
            best_model_params = {'d': d, 'c': c} if kernel_version == 'poly' else {'gamma': gamma} if kernel_version == 'rbf' else None
            best_model_llr_scores = scores
        
    save_model('svm', kernel_version, 'best_model.pkl', best_model_params, best_model_llr_scores)

    return act_dcf_values, min_dcf_values, combined_scores

def apply_eigenvalue_constraint(cov, psi):
    U, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    covNew = np.dot(U, mcol(s) * U.T)
    return covNew

def logpdf_GMM(X, gmm):
    M = len(gmm)
    N = X.shape[1]
    S = np.zeros((M, N))
    
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mcol(mu), C) + np.log(w)
    
    logdens = logsumexp(S, axis=0)
    return logdens

def llr_GMMs(X, gmm1, gmm2):
    log_dens1 = logpdf_GMM(X, gmm1)
    log_dens2 = logpdf_GMM(X, gmm2)
    return log_dens1 - log_dens2

def EM_GMM(X, gmm_init, version='full', tol=1e-6, max_iter=100, psi=1e-2):
    N = X.shape[1]
    M = len(gmm_init)
    gmm = gmm_init.copy()
    prev_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step
        S = np.zeros((M, N))
        for g in range(M):
            w, mu, C = gmm[g]
            S[g, :] = logpdf_GAU_ND(X, mcol(mu), C) + np.log(w)

        log_marginals = logsumexp(S, axis=0)
        log_responsibilities = S - log_marginals
        responsibilities = np.exp(log_responsibilities)

        # M-step
        Zg = np.sum(responsibilities, axis=1)
        Fg = np.dot(responsibilities, X.T)
        Sg = np.zeros((M, X.shape[0], X.shape[0]))

        for g in range(M):
            for i in range(N):
                xi = X[:, i].reshape(-1, 1)
                Sg[g] += responsibilities[g, i] * np.dot(xi, xi.T)

        if version == 'tied':
            overall_Sigma_new = np.zeros((X.shape[0], X.shape[0]))
            for g in range(M):
                mu_new = Fg[g] / Zg[g]
                Sigma_new = Sg[g] / Zg[g] - np.dot(mu_new.reshape(-1, 1), mu_new.reshape(1, -1))
                overall_Sigma_new += Zg[g] * Sigma_new
                gmm[g] = (Zg[g] / N, mu_new, Sigma_new)
            overall_Sigma_new /= N
            overall_Sigma_new = apply_eigenvalue_constraint(overall_Sigma_new, psi)
            for g in range(M):
                w, mu, _ = gmm[g]
                gmm[g] = (w, mu, overall_Sigma_new)
        else:
            for g in range(M):
                mu_new = Fg[g] / Zg[g]
                Sigma_new = Sg[g] / Zg[g] - np.dot(mu_new.reshape(-1, 1), mu_new.reshape(1, -1))
                if version == 'diagonal':
                    Sigma_new = np.diag(np.diag(Sigma_new))
                Sigma_new = apply_eigenvalue_constraint(Sigma_new, psi)
                w_new = Zg[g] / N
                gmm[g] = (w_new, mu_new, Sigma_new)

        log_likelihood = np.sum(log_marginals) / N
        if log_likelihood - prev_log_likelihood < tol:
            break

        prev_log_likelihood = log_likelihood

    return gmm, log_likelihood

def LBG_GMM(X, max_components=4, version='full', alpha=0.1):
    gmm = [(1.0, eval_mu(X), eval_cov(X))]
    while len(gmm) < max_components:
        new_gmm = []
        for w, mu, C in gmm:
            U, s, _ = np.linalg.svd(C)
            d = U[:, 0] * np.sqrt(s[0]) * alpha
            
            new_gmm.append((w / 2, mu - d, C))
            new_gmm.append((w / 2, mu + d, C))
        
        gmm, _ = EM_GMM(X, new_gmm, version=version)
    return gmm

def train_gmms(DTR, LTR, first_comps, second_comps, version='full'):
    class_labels = np.unique(LTR)
    gmms = {}
    for cls in class_labels:
        DTR_cls = DTR[:, LTR == cls]
        if cls == False:
            gmms[cls] = LBG_GMM(DTR_cls, max_components=first_comps, version=version)
        else:
            gmms[cls] = LBG_GMM(DTR_cls, max_components=second_comps, version=version)
    return gmms

def GMM(DTR, LTR, DVAL, LVAL, model, n_comps=[1, 2, 4, 8, 16, 32], pi=0.1, Cfn=1, Cfp=1):
    act_dcf_values = []
    min_dcf_values = []
    combined_scores = []

    permutations = list(itertools.product(n_comps, repeat=2))

    for (first_n_comp, second_n_comp) in permutations:
            gmms = train_gmms(DTR, LTR, first_n_comp, second_n_comp, model)
            llrs = llr_GMMs(DVAL, gmms[True], gmms[False])
            act_dcf = normDCF(pi, Cfn, Cfp, confMatrix(optBayesDecisions(llrs, pi, Cfn, Cfp), LVAL))
            min_dcf = minDCF(llrs, LVAL, pi, Cfn, Cfp)

            act_dcf_values.append(act_dcf)
            min_dcf_values.append(min_dcf)

             # Calculate the combined score
            combined_score = 0.3 * act_dcf + 0.7 * min_dcf
            combined_scores.append(combined_score)
            # Save model parameters and validation scores
            if combined_score == min(combined_scores):
                best_comp = {'first_class_comps': first_n_comp, 'second_class_comps': second_n_comp}
                best_llrs = llrs

    save_model('gmm', model, 'best_model.pkl', best_comp, best_llrs)

    return act_dcf_values, min_dcf_values, combined_scores

def bestClassifier(DTR, LTR, DVAL, LVAL, pi1):
    bestClassifier = {"GMM": {"Classifier Model": '', "Combined Score": float('inf')}, 
                      "LogReg": {"Classifier Model": '', "Combined Score": float('inf')}, 
                      "SVM": {"Classifier Model": '', "Combined Score": float('inf')}
                      }
    lambdas = np.logspace(-3, 2, 11)

    for classifier in ['LogReg', 'SVM', 'GMM']:
        print(f"Training {classifier} classifier...")
        if classifier == 'LogReg':
            for model in ['reduced', 'quadratic', 'centered', 'znorm', 'pca']:
                print(f"Training {model} LogReg...")
                _, _, combined_scores = LogRegression(DTR, LTR, DVAL, LVAL, lambdas, pi1, pi1, model)
                if min(combined_scores) < bestClassifier[classifier]["Combined Score"]:
                    bestClassifier[classifier] = {"Classifier Model": model, "Combined Score": min(combined_scores)}
        elif classifier == 'SVM':
            for model in ['linear', 'centered', 'poly', 'rbf']:
                print(f"Training {model} SVM...")
                if model == 'linear' or model == 'centered':
                    _, _, combined_scores = SVM(DTR, LTR, DVAL, LVAL, C_values=lambdas, kernel_version=model, params=None, pi1=pi1)
                    if min(combined_scores) < bestClassifier[classifier]["Combined Score"]:
                        bestClassifier[classifier] = {"Classifier Model": model, "Combined Score": min(combined_scores)}
                elif model == 'poly':
                    _, _, combined_scores = SVM(DTR, LTR, DVAL, LVAL, C_values=lambdas, kernel_version=model, params=[2, 1], pi1=pi1)
                    if min(combined_scores) < bestClassifier[classifier]["Combined Score"]:
                        bestClassifier[classifier] = {"Classifier Model": model, "Combined Score": min(combined_scores)}
                elif model == 'rbf':
                    for gamma in [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]:
                        _, _, combined_scores = SVM(DTR, LTR, DVAL, LVAL, lambdas, model, gamma, pi1)
                        if min(combined_scores) < bestClassifier[classifier]["Combined Score"]:
                            bestClassifier[classifier] = {"Classifier Model": model, "Combined Score": min(combined_scores)}
        else:
            for model in ['full', 'diagonal', 'tied']:
                print(f"Training {model} GMM...")
                _, _, combined_scores = GMM(DTR, LTR, DVAL, LVAL, model, pi=pi1)
                if min(combined_scores) < bestClassifier[classifier]["Combined Score"]:
                    bestClassifier[classifier] = {"Classifier Model": model, "Combined Score": min(combined_scores)}

        print(bestClassifier)

    return bestClassifier
    
def pieffvsDCFsByClassifier(LVAL, eff_prior_log_odds, classifier, model, Cfn=1, Cfp=1):
    pi_eff = 1 / (1 + np.exp(-eff_prior_log_odds))
    actual_dcf_values = []
    min_dcf_values = []

    for pi_eff_value in pi_eff:
        if classifier == 'LogReg':
            data = load_model('logreg', model, 'best_model.pkl')
        elif classifier == 'SVM':
            data = load_model('svm', model, 'best_model.pkl')
        else:
            data = load_model('gmm', model, 'best_model.pkl')
        llr = data['validation_scores']
        if classifier == 'LogReg':
            predictions = np.where(llr > -np.log(0.1 * Cfn / ((1 - 0.1) * Cfp)), True, False)
            act_dcf = normDCF(pi_eff_value, Cfn, Cfp, confMatrix(predictions, LVAL))
        else:
            act_dcf = normDCF(pi_eff_value, Cfn, Cfp, confMatrix(optBayesDecisions(llr, pi_eff_value, Cfn, Cfp), LVAL))
        min_dcf = minDCF(llr, LVAL, pi_eff_value, Cfn, Cfp)

        actual_dcf_values.append(act_dcf)
        min_dcf_values.append(min_dcf)

    return actual_dcf_values, min_dcf_values, data['model_params']

def plot_hist(D, L, fname="", flag_lda=False):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D0 = D[:, L==False]
    D1 = D[:, L==True]

    if flag_lda is True:
        plt.figure()
        plt.title(f'Histogram of {fname}')
        plt.hist(D0[0, :], bins = 10, density = True, alpha = 0.4, label = 'False')
        plt.hist(D1[0, :], bins = 10, density = True, alpha = 0.4, label = 'True')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/histograms/{fname}.png')
        plt.show()
    else:
        hFea = {
            1: 'Feature 1',
            2: 'Feature 2',
            3: 'Feature 3',
            4: 'Feature 4',
            5: 'Feature 5',
            6: 'Feature 6'
            }

        for i, feature in hFea.items():
            plt.figure()
            plt.xlabel(feature)
            plt.title(f'{feature} Histogram')
            plt.hist(D0[i-1, :], bins = 10, density = True, alpha = 0.4, label = 'False')
            plt.hist(D1[i-1, :], bins = 10, density = True, alpha = 0.4, label = 'True')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/histograms/feature{i}.png')
            plt.show()

def plot_scatter(D, L, fname="", flag_lda=False):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D0 = D[:, L==False]
    D1 = D[:, L==True]

    if flag_lda is True:
        plt.figure()
        plt.title(f'Scatter Plot of {fname}')
        plt.scatter(D0[0, :], D0[1, :], label = 'False')
        plt.scatter(D1[0, :], D1[1, :], label = 'True')

        plt.legend()
        plt.tight_layout() 
        plt.savefig(f'plots/scatters/{fname}.png')
        plt.show()

    else:
        hFea = {
            0: 'Feature 1',
            1: 'Feature 2',
            2: 'Feature 3',
            3: 'Feature 4',
            4: 'Feature 5',
            5: 'Feature 6'
            }
        
        for i in range(0, 6, 2):
            plt.figure()
            plt.xlabel(hFea[i])
            plt.ylabel(hFea[i+1])
            plt.title(f'{hFea[i]} vs {hFea[i+1]}')
            plt.scatter(D0[i, :], D0[i+1, :], label='False')
            plt.scatter(D1[i, :], D1[i+1, :], label='True')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'plots/scatters/features_{i+1}_{i+2}.png')
            plt.show()

def plot_logdens(D, L):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    for feature in [0,1,2,3,4,5]:
        plt.figure()
        for cls in [False, True]:
            D0 = D[:, L==cls]
            D1 = D0[feature, :]
            mu = eval_mu(D1, 0)
            C = eval_cov(D1, 0)
            plt.hist(D1.ravel(), bins=50, density=True, label=cls, color= 'orange' if cls == False else 'blue', alpha=0.4)
            XPlot = np.linspace(np.min(D1), np.max(D1), 1000)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_1D(vrow(XPlot), mu, C)), color='red' if cls == False else 'green', label=cls)
        plt.xlabel('Feature ' + str(feature+1))
        plt.ylabel('Density')
        plt.title('Gaussian Distr. Fit: Feature ' + str(feature+1))
        plt.legend()
        plt.savefig(f'plots/logdensity/feature_{str(feature+1)}.png')
        plt.show()

def plotBayesError(eff_prior_log_odds, dcf, mindcf, title):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure(figsize=(8, 6))
    plt.plot(eff_prior_log_odds, dcf, label=f'DCF', color='r', linestyle='-')
    plt.plot(eff_prior_log_odds, mindcf, label=f'min DCF', color='b', linestyle='-')
    plt.title(f'Bayes Error Plot ({title})')
    plt.xlabel('Log-Odds of Effective Prior (p˜)')
    plt.ylabel('Normalized DCF')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/bayes_error/{title}.png')
    plt.show()

def plotDCFsvslambda(lambdas, normdcfs, mindcfs, title):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, normdcfs, label='Actual DCF', marker='o')
    plt.plot(lambdas, mindcfs, label='Minimum DCF', marker='x')
    plt.xscale('log', base=10)
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('DCF')
    plt.title(f"DCF and MinDCF vs Lambda ({title})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/dcfs_vs_lambda/{title}.png')
    plt.show()

def plotGMMvsComponents(components, normdcfs, mindcfs, model):
    components_str = [f"{x}-{y}" for x, y in components]
    
    plt.figure(figsize=(10, 6))
    plt.plot(components_str, normdcfs, label='Actual DCF', marker='o')
    plt.plot(components_str, mindcfs, label='Minimum DCF', marker='x')
    plt.xlabel('Components (first class - second class)')
    plt.ylabel('DCF')
    plt.title(f"Actual DCF and Minimum DCF vs Components ({model})")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/gmms_vs_comps/{model}.png')
    plt.show()

def eval_plotDCFsvsCRBF(DTR, LTR, DVAL, LVAL, C_values, gamma_values):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.figure(figsize=(10, 6))
    for title, gamma in gamma_values.items():
        act_dcf_values, min_dcf_values, _ = SVM(DTR, LTR, DVAL, LVAL, C_values, 'rbf', params=gamma)
        plt.plot(C_values, act_dcf_values, label=f'Actual DCF (γ: {title})', marker='o')
        plt.plot(C_values, min_dcf_values, label=f'Minimum DCF (γ: {title})', marker='x')
    plt.xscale('log', base=10)
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title(f"DCF and MinDCF vs Lambda (RBF SVM)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/dcfs_vs_lambda/RBF.png')
    plt.show()

def eval_plotDCFsvsCRBF(DTR, LTR, DVAL, LVAL, C_values, gamma_values):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.figure(figsize=(10, 6))
    for title, gamma in gamma_values.items():
        act_dcf_values, min_dcf_values, _ = SVM(DTR, LTR, DVAL, LVAL, C_values, 'rbf', params=gamma)
        plt.plot(C_values, act_dcf_values, label=f'Actual DCF (γ: {title})', marker='o')
        plt.plot(C_values, min_dcf_values, label=f'Minimum DCF (γ: {title})', marker='x')
    plt.xscale('log', base=10)
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title(f"DCF and MinDCF vs Lambda (RBF SVM)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/dcfs_vs_lambda/RBF.png')
    plt.show()

def main():
    D, L = load_projectData()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    print("Which lab do you want to run?\n Available:\t 1, 2, 3, 4, 5, 6, 7, 8")
    choice = int(input('>\t'))

    if choice == 1:
        plot_hist(D, L)
        plot_scatter(D, L)

        for cls in [False, True]:
            Dcls = D[:, L==cls]
            mucls = mcol(Dcls.mean(1))
            Ncls = float(Dcls.shape[1])
            DCcls = Dcls - mucls
            Ccls = 1/Ncls*DCcls@DCcls.T
            varcls = Dcls.var(1)
            stdcls = Dcls.std(1)

    elif choice == 2:
        D_PCA, _ = PCA(D, 6)
        plot_hist(D_PCA, L, 'Dataset (PCA applied)')

        D_LDA, _ = LDA(D, L, [False, True])
        plot_hist(D_LDA, L, 'LDA', True)

        DTR_LDA, DVAL_LDA = LDA(DTR, LTR, [False, True], DVAL)
        error_rate_lda = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL)
        error_rate_lda_adjusted = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL, -0.019)

        for m in range(1, 7):
            DTR_PCA, DVAL_PCA = PCA(DTR, m, DVAL)
            DTR_PCALDA, DVAL_PCALDA = LDA(DTR_PCA, LTR, [False, True], DVAL_PCA)
            error_rate_pca_lda = errorRateLab3(DVAL_PCALDA, DTR_PCALDA, LTR, LVAL)

    elif choice == 3:
        plot_logdens(D, L)

    elif choice == 4:
        DTR_LDA, DVAL_LDA = LDA(DTR, LTR, [False, True], DVAL)
        error_rate_lda = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL)
        
        for version in ["gaussian", "tied", "naive"]:
            predictions, llrs = llr_binary(2, DVAL.shape[1], DTR, LTR, DVAL, version)
            predictions, llrs = llr_binary(2, DVAL.shape[1], DTR, LTR, DVAL, version)
            error_rate = np.sum(predictions != LVAL) / len(LVAL)
        
        cov_c1 = eval_cov(DTR[:, LTR == False])
        cov_c2 = eval_cov(DTR[:, LTR == True])
        corr_c1 = cov_c1 / (vcol(cov_c1.diagonal()**0.5) * vrow(cov_c1.diagonal()**0.5))
        corr_c2 = cov_c2 / (vcol(cov_c2.diagonal()**0.5) * vrow(cov_c2.diagonal()**0.5))

        DTR_PCA, DVAL_PCA = PCA(DTR, 5, DVAL)
        for version in ["gaussian", "tied", "naive"]:
            predictions, llrs = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            predictions, llrs = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            error_rate = np.sum(predictions != LVAL) / len(LVAL)

    elif choice == 5:
        effective_priors = [(pi1 / (pi1 + (1 - pi1) * Cfn / Cfp)).__round__(2) for pi1, Cfn, Cfp in [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]]
        
        DTR_PCA, DVAL_PCA = PCA(DTR, 5, DVAL)
        datasets = {
            "base": (DTR, LTR, DVAL, LVAL),
            "PCA": (DTR_PCA, LTR, DVAL_PCA, LVAL)
        }
        for dataset, (DTR_, LTR_, DVAL_, LVAL_) in datasets.items():
            for pi1, Cfn, Cfp in [(0.1, 1.0, 1.0), (0.5, 1.0, 1.0), (0.9, 1.0, 1.0)]:
                for version in ["gaussian", "tied", "naive"]:
                    predictions, llrs = llr_binary(2, DVAL_.shape[1], DTR_, LTR_, DVAL_, version)
                    DCF = normDCF(pi1, Cfn, Cfp, confMatrix(optBayesDecisions(llrs, pi1, Cfn, Cfp), LVAL_))
                    DCF_min = minDCF(llrs, LVAL_, pi1, Cfn, Cfp)
                    predictions, llrs = llr_binary(2, DVAL_.shape[1], DTR_, LTR_, DVAL_, version)
                    DCF = normDCF(pi1, Cfn, Cfp, confMatrix(optBayesDecisions(llrs, pi1, Cfn, Cfp), LVAL_))
                    DCF_min = minDCF(llrs, LVAL_, pi1, Cfn, Cfp)
                    calibration_loss = DCF - DCF_min

        best_m, best_DCF, best_min_DCF = bestmbyDCF(DTR, LTR, DVAL, LVAL, 0.1)
        DTR_PCA, DVAL_PCA = PCA(DTR, best_m, DVAL)
        logOddsRange = np.linspace(-4, 4, 50)

        for version in ["gaussian", "tied", "naive"]:
            predictions, llrs = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            dcf, mindcf = pieffvsDCFs(llrs, LVAL, logOddsRange)
            predictions, llrs = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            dcf, mindcf = pieffvsDCFs(llrs, LVAL, logOddsRange)
            plotBayesError(logOddsRange, dcf, mindcf, version)
            calibration_loss = (np.mean(dcf) - np.mean(mindcf)).round(3)

    elif choice == 6:
        lambdas = np.logspace(-4, 2, 13)
        pi_T = 0.1
        pi_emp = np.mean(LTR == 1)

        models = {
            "full": "Full Dataset",
            "reduced": "Reduced Training Samples",
            "weighted": "Prior-Weighted Linear Model",
            "quadratic": "Quadratic Model",
            "centered": "Centered Data",
            "znorm": "Z-normalized Data",
            "pca": "PCA Data"
        }

        for model, title in models.items():
            normDCF_values, minDCF_values, _ = LogRegression(DTR, LTR, DVAL, LVAL, lambdas, pi_T, pi_emp, model)
            plotDCFsvslambda(lambdas, normDCF_values, minDCF_values, title)

    elif choice == 7:
        C_values = np.logspace(-5, 0, 11)
        models = {
            "linear": ["Linear SVM", None],
            "centered": ["Centered Data SVM", None],
            "poly": ["Polynomial SVM", [2, 1]]
        }

        for model, params in models.items():
            act_dcf_values, min_dcf_values, _ = SVM(DTR, LTR, DVAL, LVAL, C_values, model, params[1])
            plotDCFsvslambda(C_values, act_dcf_values, min_dcf_values, params[0])

        gamma_values = {
            "np.exp(-4)": np.exp(-4),
            "np.exp(-3)": np.exp(-3),
            "np.exp(-2)": np.exp(-2),
            "np.exp(-1)": np.exp(-1)
            }
        eval_plotDCFsvsCRBF(DTR, LTR, DVAL, LVAL, np.logspace(-3, 2, 11), gamma_values)

    elif choice == 8:
        pi_t = 0.1
        logOddsRange = np.linspace(-4, 4, 50)
        for model in ["full", "diagonal"]:
            act_dcf_values, min_dcf_values, _ = GMM(DTR, LTR, DVAL, LVAL, model, pi=pi_t)
            plotGMMvsComponents(list(itertools.product([1,2,4,8,16,32], repeat=2)), act_dcf_values, min_dcf_values, model)
            
        bestClassifiers = bestClassifier(DTR, LTR, DVAL, LVAL, pi_t)
        
        for classifier, data in bestClassifiers.items():
            act_dcf_values, min_dcf_values, model_params = pieffvsDCFsByClassifier(LVAL, logOddsRange, classifier, data["Classifier Model"])
            if classifier == 'GMM':
                n_comp1, n_comp2 = model_params["first_class_comps"], model_params["second_class_comps"]
                plotBayesError(logOddsRange, act_dcf_values, min_dcf_values, classifier + f"_comps_{n_comp1}_{n_comp2}")
            else:
                plotBayesError(logOddsRange, act_dcf_values, min_dcf_values, classifier)
        
    else:
        print('Wrong choice')

if __name__ == '__main__':
    main()