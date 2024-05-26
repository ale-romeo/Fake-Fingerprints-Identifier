import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize as opt
import pickle
import os

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

def save_model(directory, filename, model_params, validation_scores):
    directory = f"models/{directory}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    data = {'model_params': model_params, 'validation_scores': validation_scores}
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def load_model(directory, filename):
    directory = f"models/{directory}"
    filepath = os.path.join(directory, filename)
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
    return logS[1]-logS[0]

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
    decisions = (llr > threshold).astype(int)
    return decisions

def confMatrix(predictions, labels):
    TP = np.sum((predictions == 1) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    conf_matrix = np.array([[TN, FP], [FN, TP]])
    return conf_matrix

def bayesRisk(pi1, Cfn, Cfp, conf_matrix):
    Pfn = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
    Pfp = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[0, 0])
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
    for i in range(1, len(thresholds) - 1):
        threshold = thresholds[i]
        decisions = (llr > threshold).astype(int)
        conf_matrix = confMatrix(decisions, labels)
        conf_matrix = conf_matrix.T  # Scambia righe e colonne
        dcf = normDCF(pi1, Cfn, Cfp, conf_matrix)
        min_dcf = min(min_dcf, dcf)
    
    return min_dcf

def BD_DCF_minDCF(llr, labels, pi1, Cfn = 1, Cfp = 1):
    Bayes_decisions = optBayesDecisions(llr, pi1, Cfn, Cfp)
    conf_matrix = confMatrix(Bayes_decisions, labels).T
    
    DCF = normDCF(pi1, Cfn, Cfp, conf_matrix).round(3)
    DCF_min = minDCF(llr, labels, pi1, Cfn, Cfp).round(3)
    return Bayes_decisions, DCF, DCF_min

def bestmbyDCF(DTR, LTR, DVAL, LVAL, pi1):
    best_DCF = float('inf')
    best_min_DCF = float('inf')

    for m in range(1, DTR.shape[0] + 1):
        DTR_PCA, DVAL_PCA = PCA(DTR, m, DVAL)

        best_DCF_version = float('inf')
        best_min_DCF_version = float('inf')

        for version in ["gaussian", "tied", "naive"]:
            LLR = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            BD, DCF, DCF_min = BD_DCF_minDCF(LLR, LVAL, pi1)

            if DCF < best_DCF_version:
                best_DCF_version = DCF
                best_min_DCF_version = DCF_min

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

def llrScores(D, w, b, pi_emp):
    scores = np.dot(w.T, D) + b
    llr_scores = scores - np.log(pi_emp / (1 - pi_emp))
    return llr_scores

def lambdavsDCFs(DTR, LTR, DVAL, LVAL, lambdas, pi_T, pi_emp, model, weighted=False):
    actual_dcf_values = []
    min_dcf_values = []

    for l in lambdas:
            optimal_params, _ = train_logreg(DTR, LTR, l, pi_T, weighted)
            w_opt = optimal_params[:-1]
            b_opt = optimal_params[-1]
            
            llr_scores = llrScores(DVAL, w_opt, b_opt, pi_emp)
            actual_dcf = normDCF(pi_T, 1, 1, confMatrix(optBayesDecisions(llr_scores, pi_T, 1, 1), LVAL).T)  # Assuming Cfn = Cfp = 1
            min_dcf = minDCF(llr_scores, LVAL, pi_T, 1, 1)  # Assuming Cfn = Cfp = 1
            
            actual_dcf_values.append(actual_dcf)
            min_dcf_values.append(min_dcf)
            # Save model parameters and validation scores
            model_params = {'weights': w_opt, 'bias': b_opt}
            save_model(model, f'model_lambda_{l:.1e}.pkl', model_params, llr_scores)

    return actual_dcf_values, min_dcf_values

def expand_features(X):
    n_samples, n_features = X.shape
    expanded_features = [X]  # Start with the original features
    
    # Add quadratic terms
    for i in range(n_features):
        expanded_features.append(X[:, i:i+1] ** 2)
    
    # Add interaction terms
    for i in range(n_features):
        for j in range(i+1, n_features):
            expanded_features.append(X[:, i:i+1] * X[:, j:j+1])
    
    return np.hstack(expanded_features)

def bayesError(llr, labels, eff_prior_log_odds, Cfn = 1, Cfp = 1):
    pi_eff = 1 / (1 + np.exp(-eff_prior_log_odds))
    dcf = []
    mindcf = []

    for pi_eff_value in pi_eff:
        BD, dcf_value, min_dcf_value = BD_DCF_minDCF(llr, labels, pi_eff_value, Cfn, Cfp)
        dcf.append(dcf_value)
        mindcf.append(min_dcf_value)

    return dcf, mindcf

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
    plt.xlim([-3, 3])
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

def main():
    D, L = load_projectData()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    print("Which lab do you want to run?\n Available:\t 1, 2, 3, 4, 5, 6")
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
        # Step 1: Apply PCA to the data and analyze its effects
        D_PCA, _ = PCA(D, 6)
        plot_hist(D_PCA, L, 'Dataset (PCA applied)')

        # Step 2: Apply LDA and analyze its effects
        D_LDA, _ = LDA(D, L, [False, True])
        plot_hist(D_LDA, L, 'LDA', True)

        # Step 3: Apply LDA as a classifier
        DTR_LDA, DVAL_LDA = LDA(DTR, LTR, [False, True], DVAL)
        error_rate_lda = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL)
        error_rate_lda_adjusted = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL, -0.019)

        # Step 4: Combine PCA and LDA for classification
        for m in range(1, 7):
            DTR_PCA, DVAL_PCA = PCA(DTR, m, DVAL)
            DTR_PCALDA, DVAL_PCALDA = LDA(DTR_PCA, LTR, [False, True], DVAL_PCA)
            error_rate_pca_lda = errorRateLab3(DVAL_PCALDA, DTR_PCALDA, LTR, LVAL)

    elif choice == 3:
        plot_logdens(D, L)

    elif choice == 4:
        #Apply LDA to the data and analyze its effects
        DTR_LDA, DVAL_LDA = LDA(DTR, LTR, [False, True], DVAL)
        error_rate_lda = errorRateLab3(DVAL_LDA, DTR_LDA, LTR, LVAL)
        
        for version in ["gaussian", "tied", "naive"]:
            LLR = llr_binary(2, DVAL.shape[1], DTR, LTR, DVAL, version)
            predictions = np.where(LLR >= 0, True, False)
            error_rate = np.sum(predictions != LVAL) / len(LVAL)
        
        # Extract correlation matrices
        cov_c1 = eval_cov(DTR[:, LTR == False])
        cov_c2 = eval_cov(DTR[:, LTR == True])
        corr_c1 = cov_c1 / (vcol(cov_c1.diagonal()**0.5) * vrow(cov_c1.diagonal()**0.5))
        corr_c2 = cov_c2 / (vcol(cov_c2.diagonal()**0.5) * vrow(cov_c2.diagonal()**0.5))

        DTR_PCA, DVAL_PCA = PCA(DTR, 5, DVAL)
        for version in ["gaussian", "tied", "naive"]:
            LLR = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            predictions = np.where(LLR >= 0, True, False)
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
                    LLR = llr_binary(2, DVAL_.shape[1], DTR_, LTR_, DVAL_, version)
                    BD, DCF, DCF_min = BD_DCF_minDCF(LLR, LVAL_, pi1, Cfn, Cfp)
                    calibration_loss = DCF - DCF_min

        best_m, best_DCF, best_min_DCF = bestmbyDCF(DTR, LTR, DVAL, LVAL, 0.1)
        DTR_PCA, DVAL_PCA = PCA(DTR, best_m, DVAL)
        logOddsRange = np.linspace(-4, 4, 50)

        # Compute and plot Bayes error for each model
        for version in ["gaussian", "tied", "naive"]:
            LLR = llr_binary(2, DVAL_PCA.shape[1], DTR_PCA, LTR, DVAL_PCA, version)
            dcf, mindcf = bayesError(LLR, LVAL, logOddsRange)
            plotBayesError(logOddsRange, dcf, mindcf, version)
            calibration_loss = (np.mean(dcf) - np.mean(mindcf)).round(3)

    elif choice == 6:
        lambdas = np.logspace(-4, 2, 13)
        pi_T = 0.1
        pi_emp = np.mean(LTR == 1)

        # Full Dataset - Linear Model
        normDCF_values, minDCF_values = lambdavsDCFs(DTR, LTR, DVAL, LVAL, lambdas, pi_T, pi_emp, "full")
        plotDCFsvslambda(lambdas, normDCF_values, minDCF_values, "Full Dataset")

        '''
        Observations:
        1. Significant differences for different values of λ:
        - Actual DCF increases significantly with higher λ values, especially above λ = 10^-2.
        - Minimum DCF remains constant around ~0.1 for all λ values.
        2. Effect of regularization coefficient:
        - Actual DCF:
            - Sensitive to λ. Low values of λ result in stable Actual DCF (~0.4).
            - High λ values lead to a significant increase in Actual DCF, up to ~1 for λ around 10^0.
        - Minimum DCF:
            - Not affected by different λ values, stays low (~0.1) for all λ values.
        3. Discrepancy between Actual DCF and Minimum DCF:
        - Suggests the model may not be well-calibrated for all λ values.
        - Actual DCF is with a fixed threshold, while Minimum DCF is optimized for specific thresholds.
        4. Choosing the regularization coefficient:
        - High λ leads to over-regularization, degrading performance (higher Actual DCF).
        - Low λ may not provide enough regularization, risking overfitting.
        - Intermediate λ values (around 10^-3 or 10^-2) balance regularization and performance.
        5. Model implications:
        - Calibration techniques (e.g., Platt scaling) could reduce the gap between Actual DCF and Minimum DCF.
        '''

        # Reduced Training Samples - Linear Model
        reduced_DTR = DTR[:, ::50]
        reduced_LTR = LTR[::50]
        rednormDCF_values, redminDCF_values = lambdavsDCFs(reduced_DTR, reduced_LTR, DVAL, LVAL, lambdas, pi_T, np.mean(reduced_LTR == 1), "reduced")
        plotDCFsvslambda(lambdas, rednormDCF_values, redminDCF_values, "Reduced Training Samples")
        '''
        Observations with Reduced Training Samples:
        1. Effect of Reduced Training Samples:
        - Model is more prone to overfitting with low λ values.
        - Higher λ values reduce overfitting, leading to more stable actual DCF.
        2. Actual DCF:
        - Shows more variation with different λ values when using fewer training samples.
        - Low λ values may increase actual DCF due to overfitting.
        - High λ values may stabilize or slightly increase actual DCF due to underfitting.
        3. Minimum DCF:
        - Remains relatively stable, representing the best possible performance with optimized thresholds.
        - Discrepancy between actual DCF and minimum DCF highlights the impact of regularization on model calibration.
        '''

        wnormDCF_values, wminDCF_values = lambdavsDCFs(DTR, LTR, DVAL, LVAL, lambdas, pi_T, pi_T, "weighted", weighted=True)
        plotDCFsvslambda(lambdas, wnormDCF_values, wminDCF_values, "Prior-Weighted Linear Model")
        '''
        Observations with prior-weighted logistic regression:
        For the given task, if the class distribution is balanced or the target priors are not significantly different from the observed distribution, 
        the standard logistic regression model should suffice.
        In this specific application, where the prior-weighted version appears almost identical to the non-weighted version, 
        it suggests that the class distribution might be balanced or that the priors do not significantly differ. 
        Therefore, the added complexity of using the prior-weighted model may not provide substantial benefits in this case.
        '''

        # Full Dataset - Quadratic Model
        expanded_DTR = expand_features(DTR.T).T
        expanded_DVAL = expand_features(DVAL.T).T
        qnormDCF_values, qminDCF_values = lambdavsDCFs(expanded_DTR, LTR, expanded_DVAL, LVAL, lambdas, pi_T, pi_emp, "quadratic")
        plotDCFsvslambda(lambdas, qnormDCF_values, qminDCF_values, "Quadratic Model")
        '''
        Observations for Quadratic Logistic Regression:
        1. Significant differences for different values of λ:
        - Actual DCF increases significantly for λ > 10^-2.
        - Minimum DCF remains relatively stable across different λ values, indicating consistent theoretical best performance.
        2. Effect of regularization coefficient:
        - Actual DCF:
            - Stable and low (~0.3) for small λ values (10^-4 to 10^-2).
            - Increases rapidly for λ > 10^-2, indicating over-regularization and underfitting.
        - Minimum DCF:
            - Consistently low (~0.1) with slight increase for higher λ values, showing minimal variation.
        3. Calibration discrepancy:
        - The gap between Actual DCF and Minimum DCF suggests potential calibration issues.
        - Actual DCF with a fixed threshold shows higher values, while Minimum DCF indicates optimal performance with specific thresholds.
        4. Choosing the regularization coefficient:
        - Optimal λ values are around 10^-3 to 10^-2, balancing between overfitting and underfitting.
        - High λ values degrade performance due to excessive regularization.
        5. Model implications:
        - Regularization is essential for preventing overfitting but excessive regularization leads to underfitting.
        - Calibration techniques (e.g., Platt scaling) could improve score calibration and reduce the discrepancy between Actual DCF and Minimum DCF.
        '''
        # Analyze the effects of centering on the model results
        DTR_centered, DVAL_centered = centerData(DTR, DVAL)
        centnormDCF_values, centminDCF_values = lambdavsDCFs(DTR_centered, LTR, DVAL_centered, LVAL, lambdas, pi_T, pi_emp, "centered")
        plotDCFsvslambda(lambdas, centnormDCF_values, centminDCF_values, "Centered Data")

        # Optionally, analyze the effects of Z-normalization on the model results
        DTR_zNorm, DVAL_zNorm = zNormData(DTR, DVAL)
        znormDCF_values, zminDCF_values = lambdavsDCFs(DTR_zNorm, LTR, DVAL_zNorm, LVAL, lambdas, pi_T, pi_emp, "znorm")
        plotDCFsvslambda(lambdas, znormDCF_values, zminDCF_values, "Z-normalized Data")

        # Optionally, analyze the effects of PCA on the model results (m = 5)
        DTR_PCA, DVAL_PCA = PCA(DTR, 5, DVAL)
        PCAnormDCF_values, PCAminDCF_values = lambdavsDCFs(DTR_PCA, LTR, DVAL_PCA, LVAL, lambdas, pi_T, pi_emp, "pca")
        plotDCFsvslambda(lambdas, PCAnormDCF_values, PCAminDCF_values, "PCA Data")

        models_minDCF = {
            "Full Dataset - Linear Model": minDCF_values,
            "Reduced Training Samples - Linear Model": redminDCF_values,
            "Full Dataset - Prior-Weighted Linear Model": wminDCF_values,
            "Full Dataset - Quadratic Model": qminDCF_values,
            "Centered Data - Linear Model": centminDCF_values,
            "Z-normalized Data - Linear Model": zminDCF_values,
            "PCA Data - Linear Model": PCAminDCF_values
        }

        # Print or plot the comparison of minDCF values
        for model, min_dcf in models_minDCF.items():
            print(f"{model}: {min(min_dcf).round(3)}")
        '''
        Summary of Minimum DCF Results and Analysis:

        1. Best Model(s):
        - The "Full Dataset - Quadratic Model" achieves the best results with a minimum DCF value of 0.10.

        2. Separation Rules and Distribution Assumptions:
        - Quadratic Model:
            - Assumes that the relationship between features and classes can be captured by quadratic decision boundaries.
            - Includes interaction terms and squared features to capture more complex patterns.
        - Linear Models:
            - Assume a linear decision boundary, which may not be sufficient for non-linear relationships in the data.
            - Performed worse compared to the quadratic model.

        3. Relation to Dataset Features:
        - Feature Linearity:
            - The superior performance of the quadratic model suggests the presence of non-linear relationships in the dataset.
        - Feature Scaling:
            - Z-normalized data performed better than the unnormalized linear model, indicating the importance of feature scaling.
        - Dimensionality Reduction:
            - PCA model did not perform as well, suggesting that dimensionality reduction might not capture the most discriminative features when reduced to only 5 components.

        Conclusion:
        - The quadratic model with the full dataset achieves the best results, indicating non-linear relationships in the data.
        - Feature scaling (Z-normalization) improves performance, highlighting the importance of preprocessing.
        - Dimensionality reduction using PCA may not always yield better results, depending on the data and number of components retained.
        '''

    else:
        print('Wrong choice')

if __name__ == '__main__':
    main()