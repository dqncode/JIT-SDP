library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")
library(R.matlab)
library("ggplot2")
library("RColorBrewer")
library(ggplot2)
library(png)
infpath <- "../output"  ## using LOC

outfpath <- "../results-SKEST"


#### plot methods #####
methods<- c("RF","LR","NB")
projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

criterias <- c("Erecall","Eprecision","Efmeasure","ePMI","eIFA","ePopt","cErecall","cEprecision","cEfmeasure","cPMI","cIFA","cPopt","recall","precision","accuracy","F1","Pf","G1","AUC" , "MCC")
mnames<- c("none","rum","nm","enn","tlr","rom","cnn","smo","bsmote","csmote","cenn")

gmnames<-c("None","RUM","NearMiss","ENN","Tomek Link","ROM","CNN","SMOTE","BSMOTE","SMOTE+Tomek Link","SMOTE+ENN")

titlenames=c("Recall@20%","Precision@20%","F-measure@20%","PCI@20%","IFA","Popt","Recall@20%","Precision@20%","F-measure@20%","PCI@20%","IFA","Popt","Recall","Precision","Accuracy","F-measure","Pf","G1","AUC","MCC")

lbnames<-NULL


for (method in methods) {
  for (i in seq(criterias)) {
    criteria = criterias[i]
    titlename = titlenames[i]
    cat(method, criteria, "\n")
    data <-  NULL
    sk1st <- NULL
    cnames<- paste(mnames, criteria, sep=".") 
    for (i in seq(projects)) {
      project <- projects[i]
      idata <-  NULL
      rdata <-  NULL
      for (j in seq(mnames)){
        mname <- mnames[j]
        out.fname <- sprintf(paste(infpath, "%s-%s-%s.csv", sep="/"),method,project,mname)
        idata <- read.csv(out.fname)
        # one measure
        idata<-idata[,criteria]
        rdata=cbind(rdata, idata )
        colnames(rdata)[j] <-paste(mname, criteria, sep=".")
      }
      
      rdata <- as.data.frame(rdata)
      sk <- sk_esd(rdata)
      sk1st <- rbind(sk1st, sk$groups[cnames])  
    }
    sk1st <- as.data.frame(sk1st)
    sk <- sk_esd(sk1st)
    rownames(sk$m.inf) <-cnames[sk$ord]
    ## let former subscript empty 
    rownames(sk$m.inf) <-lbnames[sk$ord]
    pdf(file=sprintf(paste(outfpath,method,"SKESD_%s_%s.pdf",  sep="/"), method,criteria), width=3.5, height=0.5)
    
    
    col1<- rainbow(8)
    col2 <- brewer.pal(8,"Dark2")
    col3 <- brewer.pal(9,"Greys")
    col4 <- brewer.pal(4,"RdBu")
    mycolor <- c(col3[9:9],col1[1:7],col2[1:8],col4[1:4])
    
    par(mfrow=c(1, 1), mai=c(0.18,0.15,0.06, 0), mex=0.1, cex=0.4)
    plot(sk,rl=FALSE, xlab=NA, ylab=NA, id.col=FALSE, yaxt="n",main="", title="", mex=0.01, cex=0.1, col=mycolor)###, col=rainbow(max(sk.1st$groups)), xlim=c(2, length(lbls)),
    text(x=seq(11),y=-0.6,srt =20, adj = c(1.0,0.1), labels = gmnames[sk$ord],xpd = TRUE,cex=0.6,col="black")
    title(main = sprintf("%s _ %s",method,titlename),cex.main =0.8,font=1,col="black",line=0.2)
    dev.off()
  }
}


