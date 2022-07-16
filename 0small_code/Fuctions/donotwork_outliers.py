                    # #IsolationForest does not work well, neither 128d nor 2d
                    # from sklearn.ensemble import IsolationForest
                    # iso = IsolationForest(contamination=0.1, warm_start=True,max_samples=40) # warm_start=f
                    # iso.fit(X_128) reduction
                    # pred = iso.predict(X_128) reduction
                    # plt.scatter(reduction[:, 0], reduction[:, 1], s=4, c=pred, alpha=0.5)
                    # plt.colorbar()
                    # plt.title(f"IsolationForest, 128")


#### show not useful for each cluster
                    # for i in range(3):
                    #     plt.subplot(321 + i)
                    #     plt.scatter(reduction[:, 0], reduction[:, 1], c=gmm.predict_proba(X_128)[:, i], cmap='viridis',
                    #                 marker='o', alpha=0.1)
                    #     value = gmm.predict_proba(X_128)[:, i]
                    #
                    #     label_check = labels[np.argmax(value)]
                    #     indices = np.where(labels == label_check)[0]
                    #     indices_value = value[indices]
                    #
                    #     number_ = int(len(indices) * th_each)
                    #     out = np.argsort(indices_value)
                    #     kk = out[:number_]  # the most th_each percent small data
                    #
                    #     #plt.scatter(reduction[indices[:], 0], reduction[indices[:], 1], c='b', marker='o', alpha=1)
                    #     plt.scatter(reduction[indices[kk], 0], reduction[indices[kk], 1], c='r', marker='o', alpha=1)
                    #     plt.title('128d GMM, 10% outliters')
                    #     plt.show()
                    

##### score value, do not work well in 128d
                    # plt.subplot(3, 2, 6)
                    # plt.scatter(reduction[:, 0], reduction[:, 1], c=gmm.score_samples(X_128), cmap='viridis', marker='x',
                    #             alpha=0.5)
                    # plt.title('Negative log-likelihood GMM, score sample')
                    # plt.colorbar()
                    # plt.show()


                    ## not good at 128 dimention try pca?
                    # for i in range(1, 5):
                    #     plt.subplot(220+i)
                    #     U = X_128
                    #     threshold = 0.95 - 0.05*i
                    #     mod = GMMOutlierDetector(n_components=args.components, threshold=threshold).fit(U)
                    #     plt.scatter(reduction[:, 0], reduction[:, 1], c=mod.predict(U), s=4, alpha=0.5)
                    #     plt.title(f"GMMOutlierDetector 128d,threshold:{round(threshold,2)},{args.components}")
                    #     plt.colorbar()
                    # plt.show()
                    # threshold = 0.95
