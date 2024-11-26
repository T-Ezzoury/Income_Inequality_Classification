library(rpart)
library(caret)
library(e1071)
library(ROCR)
library(party)
library(randomForest)
library(dplyr)
library(mlbench)
library(Rcpp)
library(solitude)
library(uwot)
library(tibble)
library(ggplot2)
library(reshape2)
library(pROC)
library(scatterplot3d)
rm(list=ls())

data= read.csv("Income_Inequality.csv", sep=";", header=TRUE)
print(paste("nombre de lignes :",nrow(data)))
print(paste("nombre de colonnes :",ncol(data)))
head(data)

#nombre d'individus de chaque classe
sum(data$Income_Inequality=="H")#nb of high inequalities
sum(data$Income_Inequality=="L")#nb of low inequalities

#country & year
length(unique(data$Country))
min(data$Year)
max(data$Year)

#valeurs manquantes
sum(is.na(data))

#Combien de pays en combien d'années
length(unique(data$Country))
length(unique(data$Year))


#Analyse statistique des données

# Sélectionner les colonnes des variables quantitatives: 
selected_columns <- data[, 4:ncol(data)]
# Calculer les moyennes des colonnes sélectionnées
moyennes <- colMeans(selected_columns, na.rm = TRUE)
# Calculer les écarts types des colonnes sélectionnées
ecart_types <- apply(selected_columns, 2, sd, na.rm = TRUE)
# Créer un data frame avec les moyennes et les écarts types
result_df <- data.frame(Moyenne = moyennes, Ecart_type = ecart_types)
view(result_df)

#boxplots
for (c in colnames(data[,-3])){
  boxplot(data[,c], main=c)
}

#longueur des colonnes
length(unique(data$Eco1)); length(unique(data$Eco2)); length(unique(data$Eco3))
length(unique(data$Energy1)); length(unique(data$Energy2)); length(unique(data$Energy3))
length(unique(data$Health1)); length(unique(data$Health2))
length(unique(data$Finan1)); length(unique(data$Finan2)); length(unique(data$Finan3)); length(unique(data$Finan4)); length(unique(data$Finan5))
length(unique(data$Governance))
length(unique(data$Poverty))
length(unique(data$Env))
length(unique(data$Other1)); length(unique(data$Other2)); length(unique(data$Other3))




#-----------------------------

#on supprime les deux premières colonnes : Country et Year
data=data[,-c(1,2)]
#on catégorise les valeurs de Income_Inequality en 1->L et 0->H
data$Income_Inequality=ifelse(data$Income_Inequality == "L", 1, 0)
data$Income_Inequality= factor(data$Income_Inequality)
View(data)

#maintenant on peut commencer l'etude: 
#1.

#Division des données en train et test
set.seed(1234)
index <- sample(1:nrow(data),round(0.70*nrow(data)))
train <- data[index,]
test <- data[-index,]
nrow(train); nrow(test)
View(test)

#-----------------------------
#arbre de decision
tree = rpart(Income_Inequality~., data=train, method="class")
summary(tree)
rpart.plot::rpart.plot(tree)
varImp(tree)
#comment sur l'importance des vars
tree$cptable

#Performances du modèle: 
#Prédiction des données test:
pr <- predict(tree, newdata = test, type = "class")
#Précision: 
mc <- table(pr, test$Income_Inequality)
mc
err= 1-((mc[1,1]+mc[2,2])/sum(mc))
print(paste("Précision :",(1-err)*100,"%"))
# sensibilite et specificite
se = mc[1,1]/(mc[1,1]+mc[2,1])
sp = mc[2,2]/(mc[1,2]+mc[2,2])
print(paste("sensibilité :",se*100,"%"))
print(paste("specificité :",sp*100,"%"))
#Pour vérifier avec les fonctions prédéfinits
se0=sensitivity(mc) 
se0
sp0=specificity(mc)
sp0
# ROC & auc
Predprob <- predict(tree, newdata = test,type = "prob")
Predprob = as.data.frame(Predprob)
Prediction <- prediction(Predprob[2],test$Income_Inequality)
performance <- performance(Prediction, "tpr","fpr")
plot(performance,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
grid()
aucDT <- performance(Prediction, measure = "auc")
aucDT <- aucDT@y.values[[1]]
print(paste("area under curve :", aucDT))
# point optimale
closest_to_01 = function(fpr, tpr){
  n=length(fpr)
  xopt=0
  yopt=0
  distance=1
  for (i in 1:n){
    if (fpr[i]*fpr[i]+(tpr[i]-1)*(tpr[i]-1)<distance*distance){
      distance= sqrt(fpr[i]*fpr[i]+(tpr[i]-1)*(tpr[i]-1))
      xopt=fpr[i]
      yopt=tpr[i]
    }
  }
  return(c(xopt, yopt))
}
fpr = attr(performance, "x.values")[[1]]
tpr = attr(performance, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr, tpr)[1], closest_to_01(fpr, tpr)[2]))



#------------------------------
#model d'arbre avec 10-fold cross validation sur train 
ctrl= trainControl(method="cv", number = 10, savePredictions = TRUE)
model1= train(Income_Inequality~., data=train, method="rpart", trControl= ctrl)
model1$method
print(model1)
varImp(model1)
summary(model1)

#Performances du modèle :

#prediction avec model1
pr1 <- predict(model1, newdata = test, type = "raw")
#Précision:
mc1 <- table(pr1, test$Income_Inequality)
mc1
err1= 1-((mc1[1,1]+mc1[2,2])/sum(mc1))
print(paste("Précision :",(1-err1)*100,"%"))
# sensibilite et specificite
sp1 = mc1[2,2]/(mc1[1,2]+mc1[2,2])
se1 = mc1[1,1]/(mc1[1,1]+mc1[2,1])
print(paste("sensibilité :",se1*100,"%"))
print(paste("specificité :",sp1*100,"%"))# ROC & auc
Predprob1 <- predict(model1, newdata = test,type = "prob")
Predprob1 = as.data.frame(Predprob1)
Prediction1 <- prediction(Predprob1[2],test$Income_Inequality)
performance1 <- performance(Prediction1, "tpr","fpr")
plot(performance1,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
grid()
aucDT1 <- performance(Prediction1, measure = "auc")
aucDT1 <- aucDT1@y.values[[1]]
print(paste("area under curve :", aucDT1))
# point optimale
fpr1 = attr(performance1, "x.values")[[1]]
tpr1 = attr(performance1, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr1, tpr1)[1], closest_to_01(fpr1, tpr1)[2]))




#------------------------------
#3. calcul d'anomalie
View(data)
data_an= data[,-1]
View(data_an)
data_an_df= as_tibble(data_an)

#Foret d'isolement: 
iso = isolationForest$new(sample_size=nrow(data_an_df))
iso$fit(data_an_df)

#score d'anomalie
anomalie_data= data_an_df%>%iso$predict()%>%arrange(desc(anomaly_score))
view(anomalie_data)

ind_max=anomalie_data$id[1:10]
ind_min=tail(anomalie_data$id, 10)
View(data[ind_max,])#individu plus anormales
View(data[ind_min,])#individus moins anormales

obs_anormales <- data_an_df[ind_max, ]
obs_normales <- data_an_df[ind_min, ]
view(obs_anormales)
view(obs_normales)

# Fusionner les deux ensembles de données
obs <- rbind(obs_anormales, obs_normales)
view(obs)
# Effectuer l'ACP sur les obs (On normalise les données)
pca_result <- prcomp(obs, scale. = TRUE) 
# Réduction de dimension - réduire à 3 dimensions
reduced_data <- predict(pca_result, newdata = obs)[, 1:3]
view(reduced_data)

# Visualisation en 3D:
s3d <- scatterplot3d(reduced_data, color = "black", pch = 19, type = "h", main = "Visualisation en 3D")
# Sélectionner les indices des observations norm et anorm
ind_anorm <- 1:10
ind_norm <- 11:20
# Visualiser les obs
s3d$points3d(reduced_data[ind_anorm, ], col = "red", pch = 19)
s3d$points3d(reduced_data[ind_norm, ], col = "blue", pch = 19)
# Ajouter des légendes pour les points
legend("topright", legend = c("Observations plus anormales", "Observations moins anormales"),
       col = c("red", "blue"), pch = 19, bty = "n", cex = 0.8)


#Cas de 50 points normales et 30 points anormales: 
ind_max=anomalie_data$id[1:30]
ind_min=tail(anomalie_data$id, 50)
obs_anormales <- data_an_df[ind_max, ]
obs_normales <- data_an_df[ind_min, ]
# Fusionner les deux ensembles de données
obs <- rbind(obs_anormales, obs_normales)
# Effectuer l'ACP sur les obs (On normalise les données)
pca_result <- prcomp(obs, scale. = TRUE) 
# Réduction de dimension - réduire à 3 dimensions
reduced_data <- predict(pca_result, newdata = obs)[, 1:3]
# Visualisation en 3D:
s3d <- scatterplot3d(reduced_data, color = "black", pch = 19, type = "h", main = "Visualisation en 3D")
# Sélectionner les indices des observations norm et anorm
ind_anorm <- 1:30
ind_norm <- 31:80
# Visualiser les obs
s3d$points3d(reduced_data[ind_anorm, ], col = "red", pch = 19)
s3d$points3d(reduced_data[ind_norm, ], col = "blue", pch = 19)
# Ajouter des légendes pour les points
legend("topright", legend = c("Observations plus anormales", "Observations moins anormales"),
       col = c("red", "blue"), pch = 19, bty = "n", cex = 0.8)



#Supprimer les 50 obserbations: 
ind_anorm=anomalie_data$id[1:50]
new_data=data[-ind_anorm,]
view(new_data)

#Diviser les données: 
set.seed(1234)
index <- sample(1:nrow(new_data),round(0.70*nrow(new_data)))
new_train <- data[index,]
new_test <- data[-index,]

#arbre de decision
new_tree = rpart(Income_Inequality~., data=new_train, method="class")
rpart.plot::rpart.plot(new_tree)
varImp(new_tree)
#CP table: 
new_tree$cptable


#Prédiction des données test:
new_pr <- predict(new_tree, newdata = new_test, type = "class")
#Précision: 
mc <- table(new_pr, new_test$Income_Inequality)
mc
err= 1-((mc[1,1]+mc[2,2])/sum(mc))
print(paste("Précision :",(1-err)*100,"%"))

# sensibilite et specificite
se = mc[1,1]/(mc[1,1]+mc[2,1])
sp = mc[2,2]/(mc[1,2]+mc[2,2])
print(paste("sensibilité :",se*100,"%"))
print(paste("specificité :",sp*100,"%"))

#Pour vérifier avec les fonctions prédéfinits
se0=sensitivity(mc) 
se0
sp0=specificity(mc)
sp0

# ROC & auc
Predprob <- predict(new_tree, newdata = new_test,type = "prob")
Predprob = as.data.frame(Predprob)
Prediction <- prediction(Predprob[2],new_test$Income_Inequality)
performance <- performance(Prediction, "tpr","fpr")
plot(performance,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
grid()
aucDT <- performance(Prediction, measure = "auc")
aucDT <- aucDT@y.values[[1]]
print(paste("area under curve :", aucDT))

#Point optimal
fpr = attr(performance, "x.values")[[1]]
tpr = attr(performance, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr, tpr)[1], closest_to_01(fpr, tpr)[2]))


#model d'arbre avec 10-fold cross validation: 
ctrl= trainControl(method="cv", number = 10, savePredictions = TRUE)
model1= train(Income_Inequality~., data=new_train, method="rpart", trControl= ctrl)
model1$method
print(model1)
varImp(model1)
summary(model1)


#prediction du modèle 1
pr1 <- predict(model1, newdata = new_test, type = "raw")
#Précision:
mc1 <- table(pr1, new_test$Income_Inequality)
mc1
err1= 1-((mc1[1,1]+mc1[2,2])/sum(mc1))
print(paste("Précision :",(1-err1)*100,"%"))

# sensibilite et specificite
sp1 = mc1[2,2]/(mc1[1,2]+mc1[2,2])
se1 = mc1[1,1]/(mc1[1,1]+mc1[2,1])
print(paste("sensibilité :",se1*100,"%"))
print(paste("specificité :",sp1*100,"%"))

# ROC & auc
Predprob1 <- predict(model1, newdata = new_test,type = "prob")
Predprob1 = as.data.frame(Predprob1)
Prediction1 <- prediction(Predprob1[2],new_test$Income_Inequality)
performance1 <- performance(Prediction1, "tpr","fpr")
plot(performance1,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")
grid()
aucDT1 <- performance(Prediction1, measure = "auc")
aucDT1 <- aucDT1@y.values[[1]]
print(paste("area under curve :", aucDT1))

# point optimale
fpr1 = attr(performance1, "x.values")[[1]]
tpr1 = attr(performance1, "y.values")[[1]]
print(paste("point optimale :", closest_to_01(fpr1, tpr1)[1], closest_to_01(fpr1, tpr1)[2]))




#..............................................
#Trouver les 10 points les plus anormales des données Test

#On supprime la colonne "Income_Inequality" 
View(test)
data_an1= test[,-1]
View(data_an1)

#On réindexe
data_an1_reindex <- data_an1 %>% tibble::as_tibble() %>% mutate(index = row_number() - 1)
view(data_an1_reindex)

#Foret d'isolement: 
iso = isolationForest$new(sample_size=nrow(data_an1_reindex))
iso$fit(data_an1_reindex)
#score d'anomalie
anomalie_data1= data_an1_reindex%>%iso$predict()%>%arrange(desc(anomaly_score))
view(anomalie_data1)

#On prend les id des 10 points les plus anormales
ind_max1=anomalie_data1$id[1:10]
print(ind_max1)
ind_points_anormales=ind_max1-1 #Pour les comparer avec les indices qui commencent par 0 dans la partie ACP et AFD
view(ind_points_anormales)  
print(ind_points_anormales)
view(data_an1_reindex[ind_max1,])#individu plus anormales


# Enregistrement des données dans un fichier CSV
write.csv(ind_points_anormales, file = "C:\\Users\\taha\\Desktop\\2A\\DATA\\UP2\\TP Mixte\\ind_points_anormales.csv", row.names = FALSE)














