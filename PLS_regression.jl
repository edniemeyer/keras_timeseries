using PLSRegressor
using DataFrames;


nfactors = 14

train = readtable("minidolar/train.csv", separator = ',')
test = readtable("minidolar/test.csv", separator = ',')

train_shift = train[:shift]
train_target = train[:f0]
train_close = train[[:v3,:v7,:v11,:v15,:v19,:v23,:v27,:v31,:v35,:v39,:v43,:v47,:v51,:v55,:v59,:v63,:v67,:v71,:v75,:v79,:v83,:v87,:v91,:v95,:v99,:v103,:v107,:v111,:v115,:v119]]


test_shift = test[:shift]
test_target = test[:f0]
test_close = test[[:v3,:v7,:v11,:v15,:v19,:v23,:v27,:v31,:v35,:v39,:v43,:v47,:v51,:v55,:v59,:v63,:v67,:v71,:v75,:v79,:v83,:v87,:v91,:v95,:v99,:v103,:v107,:v111,:v115,:v119]]

X_train = Array(train_close)
Y_train = Array(train_target)
X_test = Array(test_close)
Y_test = Array(test_target)

model          = PLSRegressor.fit(X_train,Y_train,nfactors=nfactors)
Y_pred        = PLSRegressor.predict(model,X_test)

Y_testp = Y_test + Array(test_shift)
Y_predp = Y_pred + Array(test_shift)

print("[PLS1] rmse error : $(sqrt(mean((Y_testp .- Y_predp).^2)))\n")


# nonlinear learning 
model          = PLSRegressor.fit(X_train,Y_train,nfactors=2,kernel="rbf",width=100.0)
Y_test         = PLSRegressor.predict(model,X_test)

Y_testp = Y_test + Array(test_shift)
Y_predp = Y_pred + Array(test_shift)

print("[KPLS] rmse error : $(sqrt(mean((Y_testp .- Y_predp).^2)))\n")



############### Multivariate(OHLC) PLS Regression ##################

train_ohlc = train[:, filter(x -> !(x in [:shift,:f0]), names(train))]
test_ohlc = test[:, filter(x -> !(x in [:shift,:f0]), names(test))]

X_train = Array(train_ohlc)
Y_train = Array(train_target)
X_test = Array(test_ohlc)
Y_test = Array(test_target)

model          = PLSRegressor.fit(X_train,Y_train,nfactors=nfactors)
Y_pred        = PLSRegressor.predict(model,X_test)

Y_testp = Y_test + Array(test_shift)
Y_predp = Y_pred + Array(test_shift)

print("[PLS1] rmse error : $(sqrt(mean((Y_testp .- Y_predp).^2)))\n")


# nonlinear learning
model          = PLSRegressor.fit(X_train,Y_train,nfactors=2,kernel="rbf",width=100.0)
Y_test         = PLSRegressor.predict(model,X_test)

Y_testp = Y_test + Array(test_shift)
Y_predp = Y_pred + Array(test_shift)

print("[KPLS] rmse error : $(sqrt(mean((Y_testp .- Y_predp).^2)))\n")



# min_rmse = 10
# global best_pred
# global best_w = 10
# global best_g = 10

# for g in [1,2],
#     w in linspace(0.01,3,10)
#     print(".")
#     model      = PLSRegressor.fit(X_train,Y_train,centralize=true,nfactors=g,kernel="rbf",width=w)
#     Y_pred     = PLSRegressor.predict(model,X_test)
#     rmse = sqrt(mean((Y_test .- Y_pred).^2))
#     if rmse < min_rmse
#        min_rmse = rmse
#        best_pred = Y_pred[:]
#        best_g    = g
#        best_w    = w
#    end
# end

# print("[KPLS] min mse error : $(min_rmse)")
# print("[KPLS] best factor : $(best_g)")
# print("[KPLS] best width : $(best_w)")