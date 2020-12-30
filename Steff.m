function Ste()
close all
conf_level = 0.05;
     thrAcc = 0.75;    

    imputed =readtable('data_covnet_score-imputed_missRF_increasing_1', 'PreserveVariableNames', true);
    imputed_vals = table2array(imputed);
    red = [208,28,139]/255;
    green = [184,225,134]/255;
    blue = [171,217,233]/255;
    
    [rhoP,pval] = corr(imputed_vals, 'type', 'Pearson');
    rhoP(pval>=conf_level)=0;
    rhoT = array2table(rhoP, 'VariableNames', imputed.Properties.VariableNames, 'RowNames',imputed.Properties.VariableNames)
    writetable(rhoT, 'Results.xlsx',  'Sheet', 'correlations','WriteVariableNames', true, 'WriteRowNames', true );
 
       
    [rhoS,pval] = corr(imputed_vals, 'type', 'Spearman');
    rhoS(pval>=conf_level)=0;
 
    [rhoK,pval] = corr(imputed_vals, 'type', 'Kendall');
    rhoK(pval>=conf_level)=0;
 
    fig = figure('Name', 'correlation with LABEL', 'units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    hold on; bar(1:numel(imputed.Properties.VariableNames)-1, rhoP(2:end, 1), 'FaceAlpha', 0.4, 'BarWidth', 0.1);
    hold on; bar(1+0.2:numel(imputed.Properties.VariableNames)-1+0.2,   rhoS(2:end, 1),  'FaceAlpha', 0.4,'BarWidth', 0.1);
    hold on; bar(1+0.4:numel(imputed.Properties.VariableNames)-1+0.4,   rhoK(2:end, 1),  'FaceAlpha', 0.4, 'BarWidth', 0.1);
    legend({'Pearson', 'Spearman', 'Kendall'})
    ax = gca();
    ax.XTick = 1:numel(imputed.Properties.VariableNames)-1;
    ax.XTickLabel = imputed.Properties.VariableNames(2:end);
    ax.XTickLabelRotation = 90;
    
    title('Feature correlation with LABEL')
   
    pis = zeros(1,numel(imputed.Properties.VariableNames));
   
    npos = sum(imputed.LABEL);
    nneg = sum(~imputed.LABEL);
    data_tab = imputed;
  
   
    pvals = NaN(numel(imputed.Properties.VariableNames)-1, 1)
    directions = cell(numel(imputed.Properties.VariableNames)-1, 1)
    
    numPlots = 6; 
    for startN =1: +numPlots: numel(imputed.Properties.VariableNames)
        fig = figure('Name', 'Variable distribution', 'units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); 
        for v = imputed.Properties.VariableNames(startN:(numPlots+startN-1))

            if ~strcmpi(v{1},'LABEL')
                idx = strfind(v{1},'.')
                prefix = v{1}(1:idx(1)-1)
                name_var = v{1}(idx+1:end); 
                if strcmpi(prefix, 'cat'); 
                   
                    [cs, ~,  pval_chi,~]= crosstab(imputed.LABEL, imputed.(v{1})); 

                    disp(['p-val ' v{1} ' chi square= ', num2str(pval_chi)]); 
                    pis(strcmpi(imputed.Properties.VariableNames, v))= pval_chi;
                    subplot(2,numPlots/2,find(strcmpi(imputed.Properties.VariableNames, v))-startN+1);
                    counts = cs(:,2)./[nneg; npos];
                    barh([0.75 1.75], 1-counts , 'BarWidth', 0.25, 'FaceColor', green);
                    hold on; 
                    barh([1 2], counts , 'BarWidth', 0.25, 'FaceColor', red);
                    ax = gca(); ax.XLim = [0 1.25];
                    ax.FontSize = 10;
                    ax.XTick = unique([min(counts(counts>0)), max(counts)]);
    %                 if numel(ax.XTick)>1; if (ax.XTick(1)-ax.XTick(1)<0.2); ax.XTick(1) =[]; end;end
    %                 if numel(ax.XTick)>1; ax.XTickLabel = {[num2str(round(ax.XTick(1)*100)) '%'],...
    %                                                 [num2str(round(ax.XTick(2)*100)) '%']};
    %                 else;  ax.XTickLabel = {[num2str(round(ax.XTick(1)*100)) '%']}; end
                    ax.XTickLabel ={};
                    ax.YTick=[.87 1.87];
                    ax.YTickLabel = {'Basso Rischio', 'Alto Rischio'};
                    grid on;
                    text([counts+0.02; 1-counts+0.02;], [1; 2;0.75; 1.75], ...
                        {[num2str(round(counts(1)*100)) '%'],[num2str(round(counts(2)*100)) '%'], ...
                        [num2str(round((1-counts(1))*100)) '%'],[num2str(round((1-counts(2))*100)) '%']}, ...
                        'FontSize',10);

                    title([name_var ' - p-val \approx ', num2str(round(pval_chi,4))]);
                    legend({'Absence','Presence'});
                    pvals(find(contains(imputed.Properties.VariableNames, v))-1,1) = round(pval_chi,4);
                else % non è vcategorica ma numerica
                    colonna = imputed.(v{1});
                    pval_wilcoxon_R = ranksum(colonna(imputed.LABEL==1),colonna(imputed.LABEL==0), 'tail', 'right');
                    pval_wilcoxon_L = ranksum(colonna(imputed.LABEL==1),colonna(imputed.LABEL==0), 'tail', 'left');
                    pval_wilcoxon =  min(pval_wilcoxon_R, pval_wilcoxon_L);
                    if ( pval_wilcoxon_R <  pval_wilcoxon_L); direction = "right";
                    else; direction= "less"; end
                    pis(strcmpi(imputed.Properties.VariableNames, v))= pval_wilcoxon;
                    disp(['p-val ' v{1} ' chi = ', num2str(pval_wilcoxon)]);
                    subplot(2,numPlots/2,find(strcmpi(imputed.Properties.VariableNames, v))-startN+1);
                    [N,edges] = histcounts(colonna, 'Normalization', 'probability');
                    hpos = histcounts(colonna(imputed.LABEL==1), edges, 'Normalization', 'probability');
                    hneg = histcounts(colonna(imputed.LABEL==0), edges, 'Normalization', 'probability');
                    hold on; bar(hneg, 'FaceAlpha', 0.3, 'FaceColor', 'g', 'BarWidth', 0.5);
                     hold on; bar(hpos, 'FaceAlpha', 0.3, 'FaceColor', 'r');

                    title([name_var, ' - p-val \approx ', num2str(round(pval_wilcoxon,4))]);

                    pvals(find(contains(imputed.Properties.VariableNames, v))-1, 1) =round(pval_wilcoxon,4);
                    directions{find(contains(imputed.Properties.VariableNames, v))-1, 1} =direction;
                    legend({'Basso Rischio', 'Alto Rischio'});

                end

            else
                counts = [nneg/(nneg+npos+2) (npos+2)/(nneg+npos+2)];
                subplot(2,numPlots/2,find(strcmpi(imputed.Properties.VariableNames, v))-startN+1);
                barh([1 2], counts, 'BarWidth', 0.5, 'FaceColor', blue);

                ax = gca(); ax.XLim = [0 1];
                ax.XTick = []; 
                ax.XTickLabel = {};
                ax.YTick = [1 2];
                ax.YTickLabel = {'Basso Rischio', 'Alto Rischio'};
                ax.FontSize = 10;
                grid on;
                title('Label');
                text([counts(1)+0.02; counts(2)+0.02;], [1, 2], ...
                        {[num2str(round(counts(1)*100)) '%'],[num2str(round(counts(2)*100)) '%'], ...
                        }, ...
                        'FontSize',10);
            end

        end
        %saveas(fig, ['plots_' num2str(startN) '.jpg'])
    end
    
    LABEL = table2array(imputed(:,1));
    data_tab.LABEL =[];
    perf = zeros(size(data_tab,2),6);
    for nC = 1: size(data_tab,2)
        ddC = table2array(data_tab(:,nC));
        llC = LABEL;
        pred = zeros(size(ddC));
       
        for nS = 1:size(ddC,1)
            dd = ddC; dd(nS,:) = [];
            ll = llC; ll(nS) = [];
            
            mdltree = fitctree(dd,ll ==1, 'PredictorNames', data_tab.Properties.VariableNames{nC});
            pred(nS) = predict(mdltree,ddC(nS,:));
               
        end
      
        [TPR, TNR, ACC, PPV, NPV, F1 ] = evalPredictions(pred, double(llC==1));
        perf(nC,:) = [TPR, TNR, ACC, PPV, NPV, F1 ];
        clear pred;
    end

    tab_perf = array2table([perf pvals], 'VariableNames', {'Sens', 'Spec', 'Acc', 'PPV', 'NPV', 'F1', 'p-value'}, 'RowNames', data_tab.Properties.VariableNames);
    tab_perf = [tab_perf array2table(directions, 'VariableNames', {'alternative'}, 'RowNames', data_tab.Properties.VariableNames)];
    writetable(tab_perf, 'Results.xlsx', 'Sheet', 'perfs', ...
             'WriteRowNames', true, 'WriteVariableNames', true);
         
     % sorting accuracies for feature selection    
     data_sel_tab = data_tab(:, tab_perf.Acc> thrAcc );
     disp(data_sel_tab.Properties.VariableNames);
         
  
            
    
end