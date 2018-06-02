# Aplico ranger con los mejores parámetros que anteriormente busqué

# Limpio la memoria
rm( list=ls() )
gc()

# source( "d:\\universidades\\itba\\a2018\\codigoR\\ranger\\ranger_aplicar.R" )

library( ranger )
library(randomForest)  #solo se usa para imputar nulos

# Establezco la carpeta de trabajo
setwd( "d:\\universidades\\itba\\a2018\\")

# Parámetros del dataset de entrada
karchivo_entrada       <- "adult_extendido.txt"
kcampo_clase           <- "clase"
kvalor_clase_positivo  <- ">50K"
kcampos_a_borrar       <- c( 'fnlwgt' )

# Parámetros del ranger
kformula         <- formula(paste( kcampo_clase, "~ ."))

# Parámetros de salida
kcampo_id              <- "ID_REGISTRO"
karchivo_aplicar       <- "competencia_sinclase_extendido.txt"
karchivo_prediccion    <- "salida_ranger_prediccion_global.txt"
karchivo_competencia   <- "salida_ranger_prediccion_mejores1536.txt"
karchivo_importancia   <- "ranger_importancia.txt"

# Levanto el dataset
dataset <- read.table( karchivo_entrada, header=TRUE, sep="," )

# Borro las variables que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   kcampos_a_borrar  )    ] 

# Borro los campos que no me interesan. ### Esto no hace lo mismo que la línea de arriba?
dataset <- dataset[ , !(names(dataset) %in%   kcampos_a_borrar  )    ] 

# Imputo los nulos, ya que ranger no acepta nulos
dataset <-  na.roughfix( dataset )


# Estos valores son los que generaron el mejor lift, en la búsqueda de los hiperparámetros mediante gridsearch.
vsplitrule      <-   "gini"
vmtry           <-     6
vmin.node.size  <-   100
vnum.trees      <-   500

# Genero el modelo con los mejores parámetros y usando TODO el dataset
# (Aquí no hay ni training, ni testing, ni tampoco se calcula el lift, ni AUC.)
modelo <- ranger( kformula, data = dataset, probability=TRUE, 
                  num.trees=vnum.trees,  
                  min.node.size=vmin.node.size, 
                  mtry=vmtry, 
                  splitrule=vsplitrule,
                  respect.unordered.factors=TRUE,
                  importance = 'impurity_corrected'
                )		


# Grabo la importancia de las variables
imp <- importance_pvalues( modelo, method = "janitza" )
write.table(  as.table(  cbind( rownames(imp), as.table(imp) ) ),
              , file= karchivo_importancia
              , row.names=FALSE
              , col.names=TRUE
              , quote=FALSE
              , sep="\t"
              , eol = "\r\n"
             )            


# Genero el modelo con los mejores parámetros y usando TODO el dataset
# (Aquí no hay ni training, ni testing, ni tampoco se calcula el lift, ni AUC.)
modelo <- ranger( kformula, data = dataset, probability=TRUE, 
                  num.trees=vnum.trees,  
                  min.node.size=vmin.node.size, 
                  mtry=vmtry, 
                  splitrule=vsplitrule,
                  respect.unordered.factors=TRUE
                )		

# Cargo los datos nuevos
dataset_aplicar <- read.table(  karchivo_aplicar, header=TRUE, sep="," ,  row.names=kcampo_id  )

# Borro las variables que no me interesan
dataset_aplicar <- dataset_aplicar[ , !(names(dataset_aplicar) %in%   kcampos_a_borrar  )    ] 

# Imputo los nulos, ya que ranger no acepta nulos
dataset_aplicar <-  na.roughfix( dataset_aplicar )

# Genero el vector con la predicción sobre los datos nuevos
vector_prediccion  = predict(  modelo, dataset_aplicar )

# Grabo en un archivo la predicción
tx <- data.frame( row.names( dataset_aplicar ),  vector_prediccion$predictions[ ,kvalor_clase_positivo] )
colnames( tx )  <-  c( "id", "pos_prob" )

# Ordeno por probabilidad de positivo descendente
tx_new <- tx[ order(  - tx[, "pos_prob"] ), ]
write.table( tx_new, file=karchivo_prediccion, row.names=FALSE, col.names=TRUE, quote=FALSE, sep="\t", eol = "\n")

# Grabo los  1536  con mayor probabilidad de ser postivos
write.table( tx_new$id[1:1536], file=karchivo_competencia, row.names=FALSE, col.names=FALSE, quote=FALSE, sep="\t", eol = "\n")

# Limpio la memoria
rm( list=ls() )
gc()
quit( save="no" )
