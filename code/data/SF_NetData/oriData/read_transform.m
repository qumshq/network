N =100;
% fp = fopen('BestMember_Gen=199_graph_vec','r');
for lab=1:9    
    G = zeros(N,N);
    name1 = 'N=100_100alpha=300desi_100r=';
    name2 = '_time=0';
    fno = num2str(-20 + (lab-1)*5);
    file = strcat(name1,fno,name2);
    fp = fopen(file,'r');
    for i = 1:N
        %         deg = fscanf(fp,'%d',[1,1]);
        %         for j= 1:deg
        %             node = fscanf(fp,'%d',[1,1]);
        %             node = node+1;
        %             G(i,node) = 1;
        %             G(node,i) = 1;
        %         end
        tem=fgets(fp);
        temp=zeros(1,N);
        for j=1:N
            temp(j)=str2num(tem(j));
        end
        G(i,:)=temp;
    end
    fclose(fp);
    fno_w = num2str(lab);
    w_name = '_edgelist';
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


    


