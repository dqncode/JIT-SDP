#install.packages("png")
#install.packages("effsize")
#install.packages("R.matlab")
#install.packages("ScottKnott")
library(R.matlab)
library(png)
library(effsize)
library(ScottKnott)


GetMagnitude <- function(str) {
  lbl <- NULL
  if (str == "large") {
    lbl <- "L"
  } else if (str == "medium") {
    lbl <- "M"
  } else if (str == "small") {
    lbl <- "S"
  } else if (str == "negligible") {
    lbl <- "T"
  } else {
    stop("error")
  }
  return(lbl)
}


mnames<- c("none","smo","optsmo")
gmnames=c("None","SMOTE","Optimized-SMOTE")  

infpath <- "../output-opt"  ## using LOC
outfpath <- "../results"


criterias <- c("recall","precision","F1","Pf","AUC", "MCC","ePopt","cErecall","cEprecision","cEfmeasure","cPMI","cIFA")

titlenames=c("Recall","Precision","F1","Pf","AUC","MCC","Popt","EA_Recall","EA_Precision","EA_Fmeasure","PMI","IFA")

methods<- c("RF")

projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

#main
for (method in methods){
  cat(method,  "\n")

  data.out.r <- matrix(0, nrow=length(criterias), ncol=length(mnames))
  data.out.r <- as.data.frame(data.out.r)
  rownames(data.out.r) <- criterias
  colnames(data.out.r) <- gmnames  
  
  for (i in seq(criterias)) {
    criteria <- criterias[i]
    cnames<- paste(mnames, criteria, sep=".") 
    
    for (j in seq(projects)) {
      rdata=NULL
      data <- data.all <- NULL
      project <- projects[j]
      for (k in seq(mnames)){
        mname <- mnames[k]
        out.fname <- sprintf(paste(infpath, "%s-%s-%s.csv", sep="/"), method,project,mname)
        idata <- read.csv(out.fname)
        ####one measure
        idata<-idata[,criteria]
        rdata=cbind(rdata, idata )
        colnames(rdata)[k] <-paste(mname, criteria, sep=".")
      }
      data.all[[j]] <- rdata
      data  <- rbind(data, rdata)
    }
    
    data.out.r[criteria, ] <- apply(data[, cnames], MARGIN=2, FUN=mean, na.rm=FALSE)
  }
  rownames(data.out.r) <- titlenames
  fname <-  sprintf(paste(outfpath,"mean-%s(smo-opt).csv",  sep="/"),  method)
  write.table(c("class"), file=fname, row.names=FALSE, col.names=FALSE, append=FALSE, eol=",")
  write.table(data.out.r, file=fname, row.names=TRUE,  col.names=TRUE,  append=TRUE, sep=",")

}

