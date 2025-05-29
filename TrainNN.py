import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import argparse
from typing import List

from numpy.lib.recfunctions import structured_to_unstructured


def train_model(X_train, X_val, y_train, y_val, w_train, w_val, batch_size = 256, learning_rate = 0.0005, Nodes = [100,100,100]):

    prepro_layer = keras.layers.Normalization()
    prepro_layer.adapt(X_train)
    
    print("Declaring model")
    
    model = keras.Sequential(
        [
            prepro_layer,
            keras.layers.Dense(Nodes[0], activation="relu", name="hidden1"),
            keras.layers.Dense(Nodes[1], activation="relu", name="hidden2"),
            keras.layers.Dense(Nodes[2], activation="relu", name="hidden3"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ]
    )

    print("Compiling model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    print("Beginning training")

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=1,
    )

    fit_history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size, 
        epochs=100,
        validation_data=(X_val, y_val, w_val),
        callbacks=[early_stopping_callback],
        sample_weight=w_train,
    )

    print("Training complete. Summary of model:")
    print(model.summary())

    return model



def TrainNN(variables: List,
            allfolds: bool = False,
            label: str = '',
            preload: bool = False,
            batch_size: int = 128,
            learning_rate: float = 0.0005,
            Nodes = [100,100,100]):

    if (allfolds==True): print('Running on 5 folds')
    else: print('Running on 1 fold')
    
    #Load the data, convert to unstructured.
    filedir = "/tmp/jkeller/thc_slim/"
    with h5py.File(filedir+"output_ttH_cc.h5","r") as file:
        sig_data = file["events"][:]

    with h5py.File(filedir+"output_ttH_bb.h5","r") as file:
        ttH_bb_data = file["events"][:]
    with h5py.File(filedir+"output_ttH_other.h5","r") as file:
        ttH_other_data = file["events"][:]
    with h5py.File(filedir+"output_EWK.h5","r") as file:
        EWK_data = file["events"][:]
    with h5py.File(filedir+"output_othertop.h5","r") as file:
        othertop_data = file["events"][:]
    with h5py.File(filedir+"output_ttbar.h5","r") as file:
        ttbar_data = file["events"][:]
        
    bkg_data = np.concatenate([ttH_bb_data, ttH_other_data, othertop_data, EWK_data, ttbar_data]) 
    print("Loaded sig and bkg files with {} and {} events, respectively".format(len(sig_data), len(bkg_data)))

    alldata = np.concatenate([sig_data,bkg_data])
    y = np.concatenate([np.ones(sig_data.shape[0], dtype=int), np.zeros(bkg_data.shape[0], dtype=int)])
    
    #shuffle so that it's not all signal first
    from random import shuffle
    shuffled_indices = list(range(len(alldata)))
    shuffle(shuffled_indices)
    alldata = alldata[shuffled_indices]
    y = y[shuffled_indices]

    EvNums = alldata["event"]
    EvWeights = alldata["eventWeight"]

    sum_sig = EvWeights[ y == 1 ].sum()
    sum_bkg = EvWeights[ y == 0 ].sum()

    print("Sum of weights sig = {}, bkg = {}".format(sum_sig, sum_bkg))

    #rescale so sig has same total weight
    EvWeights = EvWeights + y*EvWeights*((sum_bkg/sum_sig)-1)
    
    #Just get the variables I want
    X = alldata[variables]
    print("Training with the following variables: {}".format(X.dtype.fields.keys()))

    X = structured_to_unstructured(X)
    
    #split into train, test, validation, with 5 folds
    X_train01 = X[(EvNums % 10) > 3]
    y_train01 = y[(EvNums % 10) > 3]
    w_train01 = EvWeights[(EvNums % 10) > 3]
    X_train23 = np.concatenate([X[(EvNums % 10) < 2], X[(EvNums % 10) > 5]])
    y_train23 = np.concatenate([y[(EvNums % 10) < 2], y[(EvNums % 10) > 5]])
    w_train23 = np.concatenate([EvWeights[(EvNums % 10) < 2], EvWeights[(EvNums % 10) > 5]])
    X_train45 = np.concatenate([X[(EvNums % 10) < 4], X[(EvNums % 10) > 7]])
    y_train45 = np.concatenate([y[(EvNums % 10) < 4], y[(EvNums % 10) > 7]])
    w_train45 = np.concatenate([EvWeights[(EvNums % 10) < 4], EvWeights[(EvNums % 10) > 7]])
    X_train67 = X[(EvNums % 10) < 6]
    y_train67 = y[(EvNums % 10) < 6]
    w_train67 = EvWeights[(EvNums % 10) < 6]
    X_train89 = X[(EvNums % 10) < 8 and (EvNums % 10) > 1]
    y_train89 = y[(EvNums % 10) < 8 and (EvNums % 10) > 1]
    w_train89 = EvWeights[(EvNums % 10) < 8 and (EvNums % 10) > 1]    
    
    X_test01 = X[(EvNums % 10) < 2]
    y_test01 = y[(EvNums % 10) < 2]
    w_test01 = EvWeights[(EvNums % 10) < 2]
    X_test23 = X[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    y_test23 = y[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    w_test23 = EvWeights[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    X_test45 = X[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    y_test45 = y[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    w_test45 = EvWeights[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    X_test67 = X[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    y_test67 = y[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    w_test67 = EvWeights[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    X_test89 = X[(EvNums % 10) == 8 or (EvNums % 10) == 9]
    y_test89 = y[(EvNums % 10) == 8 or (EvNums % 10) == 9]
    w_test89 = EvWeights[(EvNums % 10) == 8 or (EvNums % 10) == 9]

    X_val01 = X[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    y_val01 = y[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    w_val01 = EvWeights[(EvNums % 10) == 2 or (EvNums % 10) == 3]
    X_val23 = X[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    y_val23 = y[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    w_val23 = EvWeights[(EvNums % 10) == 4 or (EvNums % 10) == 5]
    X_val45 = X[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    y_val45 = y[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    w_val45 = EvWeights[(EvNums % 10) == 6 or (EvNums % 10) == 7]
    X_val67 = X[(EvNums % 10) == 8 or (EvNums % 10) == 9]
    y_val67 = y[(EvNums % 10) == 8 or (EvNums % 10) == 9]
    w_val67 = EvWeights[(EvNums % 10) == 8 or (EvNums % 10) == 9]
    X_val89 = X[(EvNums % 10) < 2]
    y_val89 = y[(EvNums % 10) < 2]
    w_val89 = EvWeights[(EvNums % 10) < 2]
 
    EvNums01 = EvNums[(EvNums % 10 == 1)]
    
    print("Splitting 60/20/20 train/val/test")
    #print("In training, have {} signal and {} bkg events ({} of total and {} S/B)".format( y_train.sum() , len(y_train)-y_train.sum(), round(len(y_train)/len(y),4), round(y_train.sum()/(len(y_train)-y_train.sum()),4) ))
    #print("In val, have {} signal and {} bkg events ({} of total and {} S/B)".format( y_val.sum() , len(y_val)-y_val.sum(), round(len(y_val)/len(y),4), round(y_val.sum()/(len(y_val)-y_val.sum()),4) ))
    #print("In test, have {} signal and {} bkg events ({} of total and {} S/B)".format( y_test.sum() , len(y_test)-y_test.sum(), round(len(y_test)/len(y),4), round(y_test.sum()/(len(y_test)-y_test.sum()),4) ))

    print("Training with {}, {}, {} nodes per layer".format(Nodes[0], Nodes[1],Nodes[2]))

    if preload:
        model01 = keras.models.load_model("THC_NN01"+label)
    else:
        model01 = train_model(X_train01, X_val01, y_train01, y_val01, w_train01, w_val01, batch_size, learning_rate, Nodes)
        print('finished model 01')
        model01.save("THC_NN01"+label) 
    testout01 = model01.predict(X_test01)
    trainout01 = model01.predict(X_train01)
    
    testout = testout01
    y_test = y_test01
    w_test = w_test01
    
    trainout = trainout01
    y_train = y_train01
    w_train = w_train01
    
    if (allfolds == True):

        if preload:
            model23 = keras.models.load_model("THC_NN23"+label)
            model45 = keras.models.load_model("THC_NN45"+label)
            model67 = keras.models.load_model("THC_NN67"+label)
            model89 = keras.models.load_model("THC_NN89"+label)
        else:
            model23 = train_model(X_train23, X_val23, y_train23, y_val23, w_train23, w_val23, batch_size, learning_rate, Nodes)
            print('finished model 23')
            model23.save("THC_NN23"+label)
        
            model45 = train_model(X_train45, X_val45, y_train45, y_val45, w_train45, w_val45, batch_size, learning_rate, Nodes)
            print('finished model 45')
            model45.save("THC_NN45"+label)

            model67 = train_model(X_train67, X_val67, y_train67, y_val67, w_train67, w_val67, batch_size, learning_rate, Nodes)
            print('finished model 67')
            model67.save("THC_NN67"+label)
        
            model89 = train_model(X_train89, X_val89, y_train89, y_val89, w_train89, w_val89, batch_size, learning_rate, Nodes)
            print('finished model 89')
            model89.save("THC_NN89"+label)

        testout23 = model23.predict(X_test23)
        testout45 = model45.predict(X_test45)
        testout67 = model67.predict(X_test67)
        testout89 = model89.predict(X_test89)

        trainout23 = model23.predict(X_train23)
        trainout45 = model45.predict(X_train45)
        trainout67 = model67.predict(X_train67)
        trainout89 = model89.predict(X_train89)

        testout = np.concatenate([testout01, testout23, testout45, testout67, testout89])
        y_test = np.concatenate([y_test01, y_test23, y_test45, y_test67, y_test89])
        w_test = np.concatenate([w_test01, w_test23, w_test45, w_test67, w_test89])

        trainout = np.concatenate([trainout01, trainout23, trainout45, trainout67, trainout89])
        y_train = np.concatenate([y_train01, y_train23, y_train45, y_train67, y_train89])
        w_train = np.concatenate([w_train01, w_train23, w_train45, w_train67, w_train89])
        
    bins=100
    
    plt.figure()
    strain, _, _ = plt.hist(
        trainout[y_train.astype(bool)],
        bins=bins, density=True, histtype="step",
        color='mediumblue',
        label="Signal, train",
        weights=w_train[y_train == 1],
    )
    btrain, _, _ = plt.hist(
        trainout[~y_train.astype(bool)],
        bins=bins, density=True, histtype="step",
        color='indianred',
        label="Background, train",
        weights=w_train[y_train == 0],
    )
    stest, _, _ = plt.hist(
        testout[y_test.astype(bool)],
        bins=bins, density=True, histtype="step",
        color='midnightblue',
        label="Signal, test",
        weights=w_test[y_test == 1],
    )
    btest, _, _ = plt.hist(
        testout[~y_test.astype(bool)],
        bins=bins, density=True, histtype="step",
        color='firebrick',
        label="Background, test",
        weights=w_test[y_test == 0],
    )

    sep=0
    for i in range(0, len(stest)-1):
        sep += abs(stest[i]-btest[i])
    sep = sep/bins
    
    plt.xlabel("DNN output")
    plt.ylabel("Events")
    plt.legend(frameon=False)
    plt.figtext(0.14,0.89,'separation = {}'.format(round(sep,3)))
    plt.savefig("NN_output"+label+".png", format='png')

    #overtraining indices
    overtrain_sig = 0
    overtrain_bkg = 0
    for i in range(0,bins):
        overtrain_sig += abs(stest[i] - strain[i])/bins
        overtrain_bkg += abs(btest[i] - btrain[i])/bins

    print('overtraining signal = ',overtrain_sig)
    print('overtraining backgd = ',overtrain_bkg)

    #ROC
    integral_sig = sum(stest[0:bins])
    integral_bkg = sum(btest[0:bins])

    nsignal = np.zeros(bins)
    nbackgd = np.zeros(bins)
    for i in range(0,bins):
        nsignal[i] = sum(stest[i:bins])
        nbackgd[i] = sum(btest[i:bins])
        print('{} {} {}'.format(i,nbackgd[i]/integral_bkg, nsignal[i]/integral_sig))

    plt.figure()
    plt.plot(nbackgd/integral_bkg, nsignal/integral_sig,'o-',color='blueviolet')
    plt.xlabel('Background efficiency',fontsize=12)
    plt.ylabel('Signal efficiency',fontsize=12)
    plt.savefig('ROC'+label+'.png',format='png')

    auc = 0
    for i in range(0, bins-1):
        auc+= (nbackgd[i] - nbackgd[i+1])*(nsignal[i] + nsignal[i+1])/(2*integral_sig*integral_bkg)
    print("ROC AUC = {}".format(auc))

    return (auc, overtrain_sig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--allfolds', dest='allfolds', action = 'store_true', help = 'If true, do all five folds. If false, just 1')
    parser.add_argument('-s', '--scheme', dest = 'scheme', default = '1',
                        help = 'flavor tagging scheme')
    parser.add_argument('-p', '--preload', dest = 'preload', action = 'store_true',
                        help = 'load already trained models')
    parser.add_argument('-v', '--doVars', dest = 'doVars', action = 'store_true',
                        help = 'Do variable removal study')
    parser.add_argument('-o', '--doOpt', dest = 'doOpt', action = 'store_true',
                        help = 'Optimize batch sizes and learning rates')
    parser.add_argument('-n', '--doNodes', dest = 'doNodes', action = 'store_true',
                        help = 'Do study of number of Nodes per layer')    
    
    args = parser.parse_args()

    variables = ["met",
                 "metphi",
                 "ptl1",
                 "etal1",
                 "phil1",
                 "nelectrons",
                 "njets",
                 "pta1",
                 "pta2",
                 "pta3",
                 "pta4",
                 "pta5",
                 "pta6",
                 "pta7",
                 "pta8",
                 "etaa1",
                 "etaa2",
                 "etaa3",
                 "etaa4",
                 "etaa5",
                 "etaa6",
                 "etaa7",
                 "etaa8",
                 "phia1",
                 "phia2",
                 "phia3",
                 "phia4",
                 "phia5",
                 "phia6",
                 "phia7",
                 "phia8",
                 "ea1",
                 "ea2",
                 "ea3",
                 "ea4",
                 "ea5",
                 "ea6",
                 "ea7",
                 "ea8",
                 ]
    
    if (args.scheme=='1'): variables.extend(["tagbinscheme1a1","tagbinscheme1a2","tagbinscheme1a3","tagbinscheme1a4","tagbinscheme1a5","tagbinscheme1a6","tagbinscheme1a7","tagbinscheme1a8"])
    if (args.scheme=='2'): variables.extend(["tagbinscheme2a1","tagbinscheme2a2","tagbinscheme2a3","tagbinscheme2a4","tagbinscheme2a5","tagbinscheme2a6","tagbinscheme2a7","tagbinscheme2a8"])
    if (args.scheme=='3'): variables.extend(["tagbinscheme3a1","tagbinscheme3a2","tagbinscheme3a3","tagbinscheme3a4","tagbinscheme3a5","tagbinscheme3a6","tagbinscheme3a7","tagbinscheme3a8"])
    if (args.scheme=='4'): variables.extend(["tagbinscheme4a1","tagbinscheme4a2","tagbinscheme4a3","tagbinscheme4a4","tagbinscheme4a5","tagbinscheme4a6","tagbinscheme4a7","tagbinscheme4a8"])
    if (args.scheme=='5'): variables.extend(["tagbinscheme5a1","tagbinscheme5a2","tagbinscheme5a3","tagbinscheme5a4","tagbinscheme5a5","tagbinscheme5a6","tagbinscheme5a7","tagbinscheme5a8"])
    if (args.scheme=='6'): variables.extend(["tagbinscheme6a1","tagbinscheme6a2","tagbinscheme6a3","tagbinscheme6a4","tagbinscheme6a5","tagbinscheme6a6","tagbinscheme6a7","tagbinscheme6a8"])
    if (args.scheme=='7'): variables.extend(["tagbinscheme7a1","tagbinscheme7a2","tagbinscheme7a3","tagbinscheme7a4","tagbinscheme7a5","tagbinscheme7a6","tagbinscheme7a7","tagbinscheme7a8"])
    if (args.scheme=='8'): variables.extend(["dba1","dba2","dba3","dba4","dba5","dba6","dba7","dba8","dca1","dca2","dca3","dca4","dca5","dca6","dca7","dca8"])
    if (args.scheme=='9'): variables.extend(["pba1","pba2","pba3","pba4","pba5","pba6","pba7","pba8","pca1","pca2","pca3","pca4","pca5","pca6","pca7","pca8","pua1","pua2","pua3","pua4","pua5","pua6","pua7","pua8","ptaua1","ptaua2","ptaua3","ptaua4","ptaua5","ptaua6","ptaua7","ptaua8"])

    
    if(args.doVars == False and args.doOpt == False and args.doNodes == False): output = TrainNN(variables = variables, allfolds = args.allfolds, label = 'scheme'+args.scheme, preload = args.preload)
    
    if(args.doOpt == True):
        batch_sizes = [512,256,128,64,32]
        learning_rates = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

        batch_rocs = np.zeros( len(batch_sizes) )
        rate_rocs = np.zeros( len(learning_rates) )

        bestpoint = 0

        for i in range(len(batch_sizes)):
            print('Training using batch_size = {}'.format(batch_sizes[i]))
            output = TrainNN(variables = variables, allfolds = False, batch_size = batch_sizes[i])
            if (output[0] > np.max(batch_rocs)): bestpoint = i
            batch_rocs[i] = output[0]
        
        print('ROC optimization of batch sizes')
        for i in range(len(batch_sizes)):
            print("{}    {}".format(batch_sizes[i], round(batch_rocs[i],5)))
        
        opt_batch_size = batch_sizes[bestpoint] 
        print('Choosing best batch size {}'.format(opt_batch_size)) 

        bestpoint = 0
        for i in range(len(learning_rates)):
            print('Training using learning_rate = {}'.format(learning_rates[i]))
            output = TrainNN(variables = variables, allfolds = False, batch_size = opt_batch_size, learning_rate = learning_rates[i])
            if (output[0] > np.max(rate_rocs)): bestpoint = i
            rate_rocs[i] = output[0]

        print('ROC optimization of learning rates')
        for i in range(len(learning_rates)):
            print("{}    {}".format(learning_rates[i], round(rate_rocs[i],5)))

        opt_learning_rate = learning_rates[bestpoint]
        print('Best learning rate {}'.format(opt_learning_rate))
        print('Best batch size {}'.format(opt_batch_size))
        print('Best ROC achieved = {}'.format(round(np.max(rate_rocs),4)))

    if(args.doVars == True):

        output = TrainNN(variables = variables, allfolds = False, batch_size = 128)
        baseroc = output[0]
        var_rocs = np.zeros(len(variables))

        for i in range(len(variables)):
            testvars = variables[0:i] + variables[i+1:len(variables)] 
            print('skipping variable '+variables[i])
            output = TrainNN(variables = testvars, allfolds = False, batch_size = 128)
            var_rocs[i] = output[0]

        print('Test of variable removal')
        print('All variables, AUC = {}'.format(baseroc))
        for i in range(len(variables)):
            print(' Remove {:12}: AUC = {:<7} ({})'.format(variables[i],round(var_rocs[i],4),round(var_rocs[i] - baseroc,4)))


    if(args.doNodes == True):

        Nodelist = ([25,15,10], [40,20,10], [50,40,25], [50,50,50], [100,40,25], [100,50,50], [100,80,60], [100,100,100])
        node_rocs = np.zeros(len(Nodelist))
        node_ovts = np.zeros(len(Nodelist))
        
        for i in range(len(Nodelist)):
            print('Training with layers with {}, {}, {} nodes'.format(Nodelist[i][0],Nodelist[i][1],Nodelist[i][2]))
            output = TrainNN(variables = variables, allfolds = False, batch_size = 128, learning_rate = 0.0005, Nodes = Nodelist[i]) 
            node_rocs[i] = output[0]
            node_ovts[i] = output[1]

        print('Study of N nodes per layer')
        for i in range(len(Nodelist)):
            print('{}, {}, {} nodes: ROC = {}, overtraining = {}'.format(Nodelist[i][0],Nodelist[i][1],Nodelist[i][2],round(node_rocs[i],4),round(node_ovts[i],3)))
