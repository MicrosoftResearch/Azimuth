#Supplementary Figure 3  |  Source code for assigning a score to a hypothetical deletion 
#pattern associated with microhomology 
# ------------------------------------------
# comes from the Supplementary Info of the paper, in pdf form, copied here, but refactored to make a function
#    rather than to write it to file
# also see their web server version: http://www.rgenome.net/mich-calculator/ where they say:
# Insert one or more query sequences (A, G, T, C only) flanking the same length at a cleavage site (100bp or less, 60~80bp recommended).

from math import exp       
from re import findall 
 
def compute_score(seq, tmpfile1="1.before removing duplication.txt", tmpfile2="2.all microhomology patterns.txt", verbose=False):
    length_weight=20.0 
    left=30        # Insert the position expected to be broken. 
    right=len(seq)-int(left) 
    #print 'length of seq = '+str(len(seq)) 
     
    file_temp=open(tmpfile1, "w") 
    for k in range(2,left)[::-1]: 
            for j in range(left,left+right-k+1): 
                    for i in range(0,left-k+1): 
                            if seq[i:i+k]==seq[j:j+k]: 
                                    length=j-i 
                                    file_temp.write(seq[i:i+k]+'\t'+str(i)+'\t'+str(i+k)+'\t'+str(j)+'\t'+str(j+k)+'\t'+str(length)+'\n') 
    file_temp.close() 
     
    ### After searching out all microhomology patterns, duplication should be removed!! 
    f1=open(tmpfile1, "r") 
    s1=f1.read() 
     
    f2=open(tmpfile2, "w") #After removing duplication 
    f2.write(seq+'\t'+'microhomology\t'+'deletion length\t'+'score of a pattern\n') 
     
    if s1!="": 
            list_f1=s1.strip().split('\n') 
            sum_score_3=0 
            sum_score_not_3=0 
     
            for i in range(len(list_f1)): 
                    n=0 
                    score_3=0 
                    score_not_3=0 
                    line=list_f1[i].split('\t') 
                    scrap=line[0] 
                    left_start=int(line[1]) 
                    left_end=int(line[2]) 
                    right_start=int(line[3]) 
                    right_end=int(line[4]) 
                    length=int(line[5]) 
     
                    for j in range(i): 
                            line_ref=list_f1[j].split('\t') 
                            left_start_ref=int(line_ref[1]) 
                            left_end_ref=int(line_ref[2]) 
                            right_start_ref=int(line_ref[3]) 
                            right_end_ref=int(line_ref[4]) 
     
                            if (left_start >= left_start_ref) and (left_end <= left_end_ref) and (right_start >= right_start_ref) and (right_end <= right_end_ref): 
                                    if (left_start - left_start_ref)==(right_start - right_start_ref) and (left_end - left_end_ref)==(right_end - right_end_ref): 
                                            n+=1 
                            else: pass 
                           
                    if n == 0: 
                            if (length % 3)==0: 
                                    length_factor = round(1/exp((length)/(length_weight)),3) 
                                    num_GC=len(findall('G',scrap))+len(findall('C',scrap)) 
                                    score_3=100*length_factor*((len(scrap)-num_GC)+(num_GC*2)) 
                                     
                            elif (length % 3)!=0: 
                                    length_factor = round(1/exp((length)/(length_weight)),3) 
                                    num_GC=len(findall('G',scrap))+len(findall('C',scrap)) 
                                    score_not_3=100*length_factor*((len(scrap)-num_GC)+(num_GC*2)) 
     
                            f2.write(seq[0:left_end]+'-'*length+seq[right_end:]+'\t'+scrap+'\t'+str(length)+'\t'+str(100*length_factor*((len(scrap)-num_GC)+(num_GC*2)))+'\n') 
                    sum_score_3+=score_3 
                    sum_score_not_3+=score_not_3 
     
            mh_score = sum_score_3+sum_score_not_3
            oof_score = (sum_score_not_3)*100/(sum_score_3+sum_score_not_3)
            if verbose:
                print 'Microhomology score = ' + str(mh_score) 
                print 'Out-of-frame score = ' + str(oof_score) 
    f1.close() 
    f2.close()
    return mh_score, oof_score

if __name__ == '__main__':
    seq='GGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCG'    # The length of sequence is recommend within 60~80 bases. 

    tmpfile1 = "1.before removing duplication.txt"
    tmpfile2 = "2.all microhomology patterns.txt"
    
    mh_score, oof_score = compute_score(seq, tmpfile1=tmpfile1, tmpfile2=tmpfile2, verbose=True)

    # The row of output file is consist of (full sequence, microhomology scrap, deletion length, score of pattern). 

    #correct output is
    #Microhomology score = 4662.9
    #Out-of-frame score = 50.7473889639
    #GGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCG    
    
    print seq  