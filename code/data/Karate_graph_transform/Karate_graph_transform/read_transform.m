N = 34;
% fp = fopen('BestMember_Gen=199_graph_vec','r');
for lab=1:10    
    G = zeros(N,N);
    name1 = 'BestMember_Gen=';
    name2 = '_graph_vec';
    fno = num2str((lab-1)*5);
    file = strcat(name1,fno,name2);
    fp = fopen(file,'r');
    for i = 1:N
        deg = fscanf(fp,'%d',[1,1]);
        for j= 1:deg
            node = fscanf(fp,'%d',[1,1]);
            node = node+1;
            G(i,node) = 1;
            G(node,i) = 1;
        end
    end
    fclose(fp);
    fno_w = num2str(lab);
    w_name = 'change.edgelist';
    file_w = strcat(fno_w,w_name);
    fp = fopen(file_w,'w');
    for i = 1:N
        for j= i:N
            if G(i,j)
                fprintf(fp,'%d %d\n',i-1,j-1);
            end
        end
    end
    fclose(fp);
end


    


