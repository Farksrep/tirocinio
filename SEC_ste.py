#import
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, matthews_corrcoef, cohen_kappa_score
from sklearn import metrics
import numpy as np
from collections import Counter
import random


class SEC_ste (object):
    def __init__(self, X, Y, features, feature_selector, data_aug_method, learning_alg, n_fold, n_it ):
        self.X = X
        self.Y=Y
        self.features=features
        self.feature_selector = feature_selector
        self.data_aug_method = data_aug_method
        self.learning_alg = learning_alg
        self.n_fold = n_fold
        self.n_it = n_it
        
        self.spec=np.zeros(self.n_it)
        self.sens = np.zeros(self.n_it)
        self.f1=np.zeros(self.n_it)
        self.acc=np.zeros(self.n_it)
        self.prec=np.zeros(self.n_it)
        self.matt =np.zeros(self.n_it)
        self.cohen=np.zeros(self.n_it)  
        
    def specificity_score(self, ypred,y):
        tn, fp, fn, tp = confusion_matrix(ypred, y).ravel()
    
        return tn/(tn+fp)

    
    
    def undersampler(self, X_train_selected, y_train):
        rus = RandomUnderSampler(random_state=0)
        #benchè non sia un aumento, ma una riduzione, viene più comodo per come è costruito il tutto lasciar il nome invariato
        X_train_selected_aug, y_train_aug = rus.fit_resample(X_train_selected, y_train)
        
        return X_train_selected_aug, y_train_aug
        
    #data augmentation se non vengono scelte variabili categoriche (Quindi usa SMOTE)
    def SMOTE_augmentation(self, X_train_selected, y_train):
    
        sm=SMOTE(random_state=41)
        X_train_selected_aug, y_train_aug = sm.fit_resample(X_train_selected, y_train)

        return X_train_selected_aug, y_train_aug
    
    #data augmentation se vengono scelte le variabili categoriche (Quindi usa SMOTENC)
    def SMOTENC_augmentation_cat(self, X_train_selected, y_train, cat_indexes):
    
        sm=SMOTENC(random_state=41, categorical_features=cat_indexes)
        X_train_selected_aug, y_train_aug = sm.fit_resample(X_train_selected, y_train)

        return X_train_selected_aug, y_train_aug
    
    #data augmentation con oversample
    def oversample_augmentation(self, X_train_selected, y_train, features_selected, err_perc, prob_modifica): #err_perc è la percentuale di quelli che avranno l'errore nella data aug.
        #conta quante volte compare 1 e quante 0 nell'elemtno passato tra ().
        counter = Counter(y_train) 
   
        #così so quanti sono gli elementi a cui dovrò mettere un errore.
        missing=counter[0]-counter[1]
 
        #creo un indice da cui so che i successivi elementi sono stati aggiunti
        partenza=counter[0]+counter[1] 
   
        #prendo il valore min e max per ogni feature
        min_feat=X_train_selected.min(axis=1) 
        max_feat=X_train_selected.max(axis=1) 
        
        #prendo la metà dell'intervallo per ogni feat. Se mod_prob=1 allora shifto di mezzo intervallo.
        scale=(max_feat-min_feat)/2 
        
        #verifico la percentuale inserita nell'argomento
        if  err_perc>1:
            err_perc=err_perc/100
        
        #pareggio le due classi
        oversample = RandomOverSampler(sampling_strategy='minority')
        
        X_train_selected_aug, y_train_aug = oversample.fit_resample(X_train_selected, y_train)
        
        #numero di quelli che avranno l'errore modificare (quelli che voglio aggiungere*%). Potrebbe non essere intero
        n_modificate=missing*err_perc
        #ora è intero: round approssima all'intero più vicino
        n_modificate=round(n_modificate) 
        #andarò a prendere n_modificate alla fine di X_train_selected_aug, andando a vedere le ultime "missing".
        
        #mi salvo gli indici dei valori a cui devo andare ad aggiugere l'errore (l'errore viene aggiunto solo ai valori "nuovi")
        indici_prescelti= random.sample(range(partenza, partenza+missing), n_modificate)
        
        
        #matrice che mi contiene per ogni indice_prescelto un valore tra 0  e 1 che mi indica se verrà modificato e di quanto
       
        prob_mod = np.random.sample((n_modificate, len(features_selected)))
        mu,sigma=0, 0.1
    #itero indici e features. E vado poi a verificare quanto vale prob_mod per vedere se il valore viene modificato e, se si, di quanto
    #quindi scorro le righe e dopo le colonne       
        for i, ind in enumerate(indici_prescelti) :
            
            for j, feature in enumerate(features_selected):
                
                #tre casi per le possibili feature che posso trovare
                if abs(prob_mod[i,j])< prob_modifica:
                    if feature.startswith('CAT'):
                        
                        #vado a calcolare il modulo della probabilità di modifica: se è >sigma (68%?) allora ne inverto il valore
                        if abs(prob_mod[i,j])< prob_modifica:
                            if X_train_selected_aug[ind, j]==0:
                                X_train_selected_aug[ind, j]==1
                            else:
                                X_train_selected_aug[ind, j]==0
                               
                    elif feature.startswith('NUM'):
                        #vado a modificare il dato. Se prob_mod=0 il dato rimane invariato
                        X_train_selected_aug[ind, j]= X_train_selected_aug[ind, j] + scale[j]*np.random.normal(mu,sigma)
                        #verifico che se fosse uscito dall'intervallo gli imposto il valore massimo
                        X_train_selected_aug[ind, j]=min(X_train_selected_aug[ind, j], max_feat[j])
                        #verifico che se fosse uscito dall'intervallo gli imposto il valore minimo
                        X_train_selected_aug[ind, j]=max(X_train_selected_aug[ind, j], min_feat[j])
                    else :
                        #vado a modificare il dato. 
                        X_train_selected_aug[ind, j]= round(X_train_selected_aug[ind, j] + scale[j]*np.random.normal(mu,sigma))
                        #verifico che se fosse uscito dall'intervallo gli imposto il valore massimo
                        X_train_selected_aug[ind, j]=min(X_train_selected_aug[ind, j], max_feat[j])
                        #verifico che se fosse uscito dall'intervallo gli imposto il valore minimo
                        X_train_selected_aug[ind, j]=max(X_train_selected_aug[ind, j], min_feat[j])
                
            
        return X_train_selected_aug, y_train_aug

    def fit_score(self):
        #variabili
        parzial_features=list()
        y_pred=np.zeros(self.Y.shape[0])
        
        
        #n iterazioni
        for niter in range(self.n_it):
            skf = StratifiedKFold(n_splits=self.n_fold)
            
            #outerfold
            for train_index, test_index in skf.split(self.X, self.Y): 

                # mi creo i punti di train (270) e di test (30) ad ogni iterazione cambiano 
                X_train, X_test = self.X[train_index,:], self.X[test_index,:]
                #creo le label del train e quelle del test (monodimensionale lungo 270 e 30)
                y_train, y_test = self.Y[train_index], self.Y[test_index]
                
#FEATURE SELECTION
                self.feature_selector.fit(X_train, y_train)
                indexes = np.where(self.feature_selector.support_ == True)

                #quindi qui ho 270x20 e 30x20        
                X_train_selected=self.feature_selector.transform(X_train)
                #X_train_selected=X_train[:,indexes]
                X_test_selected=self.feature_selector.transform(X_test)
                
                indici=[]
                for x in np.nditer(indexes):
                    parzial_features.append(self.features[x])
                    indici.append(int(x))
                features_selected = self.features[indici] 
# DATA AUGMENTATION
                
                if self.data_aug_method=='smote':
        
                    #array che viene creato ogni volta per 
                    cat_indexes = []
                    #divido in categorici e numerici. i lo uso come indice, feature mi indica la i-esima feature 
                   
                    #fa passare da 42 a 20 features
                    for i, feature in enumerate(features_selected): #Itero con indice e feature solo sulle selezionate
                        if feature.startswith('CAT'):
                            cat_indexes.append(i)
                        #Se viene scelta qualche feature categorica implemento SMOTENC altrimenti SMOTE
                    if cat_indexes:
                        X_train_selected_aug, y_train_aug = self.SMOTENC_augmentation_cat(X_train_selected, y_train, cat_indexes)
                      
                    else:
                        X_train_selected_aug, y_train_aug = self.SMOTE_augmentation(X_train_selected, y_train)
                      
                elif self.data_aug_method=='oversample':
                    X_train_selected_aug, y_train_aug = self.oversample_augmentation(X_train_selected, y_train, features_selected, err_perc=10, prob_modifica=0.7)
             
                    
                    
                elif self.data_aug_method=='undersample':
                     X_train_selected_aug, y_train_aug = self.undersampler(X_train_selected, y_train)
                     counter = Counter(y_train_aug)
           
                                   
                elif self.data_aug_method=='none':
                    X_train_selected_aug, y_train_aug = X_train_selected, y_train
                    #return 
##TRAIN
                clf = self.learning_alg.fit(X_train_selected_aug, y_train_aug)
    
    
##TEST
                y_pred[test_index] = clf.predict(X_test_selected)
               




        #fine ciclo for train_index, test_index in skf.split(X, Y):

        # QUI ho le predizioni ottenute con questo split del dataset
        #mi salvo il valore di ogni indice statistico, nell'array. Dovrò farne la media.
            self.f1[niter]=f1_score(y_pred,self.Y)
            self.sens[niter] = recall_score(y_pred,self.Y)
            self.acc[niter]=accuracy_score(y_pred,self.Y)
            self.prec[niter]=precision_score(y_pred,self.Y)
            self.matt[niter]=matthews_corrcoef(y_pred,self.Y)
            self.cohen[niter]=cohen_kappa_score(y_pred,self.Y)
            self.spec[niter]=self.specificity_score(y_pred,self.Y)
            
        return parzial_features

    def print_result(self):
        print("f1: %f\n" %(np.mean(self.f1)))
        print("Sensibilità: %f\n" %(np.mean(self.sens)))
        print("Accuratezza: %f\n" %(np.mean(self.acc)))
        print("Precisione: %f\n" %(np.mean(self.prec)))
        print("Matthews: %f\n" %(np.mean(self.matt)))
        print("Kappa di Cohen: %f\n" %(np.mean(self.cohen)))
        print("Specificità: %f\n" %(np.mean(self.spec)))
   