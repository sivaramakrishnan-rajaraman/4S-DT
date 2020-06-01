function [all_ACC,all_sn, all_sp] = ConfusionMat_MultiClass(cmat,classNames)

classNum=numel(classNames);
ACC_Class=zeros(classNum,1);       % Accuracy
SN_Class=zeros(classNum,1);        % Sensitivity
SP_Class=zeros(classNum,1);        % Specificity


for C=1:classNum

    TP=0;   TN=0;   FP=0;  FN=0;
    
 %%%%%%%%%%%%%%%%% compute TP

            TP =TP +cmat(C,C);
                

%%%%%%%%%%%%%%%%% compute FN
             i=C;
                 for j=1:classNum
                     if j ~= i 
                      FN =FN +cmat(i,j);
                     end
                 end
            

%%%%%%%%%%%%%%%%% compute FP
             i=C;
                 for j=1:classNum
                     if j ~= i
                       FP =FP +cmat(j,i);
                     end
                 end
           
%%%%%%%%%%%%%%%%% compute TN
            for i=1:classNum
                if i ~= C
                   for j=1:classNum
                        if j ~= C
                             TN= TN +cmat(i,j);
                        end
                   end
                end
            end
 

ACC_Class(C,1)=(TP+TN)/(TP+TN+FP+FN);
SN_Class(C,1) = TP / (TP + FN);        
SP_Class(C,1)= TN /(TN + FP); 

            
end
    

all_ACC= mean(ACC_Class) ;
all_sn= mean(SN_Class) ;
all_sp= mean(SP_Class) ;


end
