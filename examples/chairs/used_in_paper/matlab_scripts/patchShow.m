function [I meta] = patchShow(P, varargin)
   
%PATCHSHOW displays patches
%   PATCHSHOW(P, ...) displays the N patches stored in the HxWxN or HxWx3xN matrix P.
%   Alternatively, P can be a cell array.
%
%   PATCHSHOW(P, 'Rows', r) displays the patches in r rows.
%
%   PATCHSHOW(P, 'Cols', c) displays the patches in c cols.
%
%   PATCHSHOW(P, 'NumPatches', N) only displays the first N patches.
%
%   PATCHSHOW(P, 'LineWidth', s) sets spacing between patches to s.
%
%   PATCHSHOW(P, 'BgColor', g) sets the background brightness to g, 0 <= g <= 1.
%
%   PATCHSHOW(P, 'CLim', [cLow cHigh]) specifies the range of color coded values.
%
%   PATCHSHOW(P, 'CLimPerc', [cLowPrc cHighPrc]) will lead to the lower cLowPrc and
%   upper cHighPrc percent of values being displayed in black or white, respectively.
%
%   PATCHSHOW(XXX, 'Color') shows RGB images; needs the input array to have third dimension equal to 3.
%
%   PATCHSHOW(XXX, 'NoShow') does not show the resulting image, just produces it as an output array.

% Written by Lukas Theis
% Modified by Alexey Dosovitskiy

if nnz(strcmp(varargin, 'Color')) > 0
    varargin{find(strcmp(varargin, 'Color'))} = 'NoShow';
    if numel(find(strcmp(varargin, 'CLim'))) == 0 && numel(find(strcmp(varargin, 'clim'))) == 0 && numel(find(strcmp(varargin, 'cLim'))) == 0
        varargin{end+1} = 'CLim';
        varargin{end+1} = [min(P(:)) max(P(:))];
    end
    for channel = 1:3
        [I(:,:,channel) meta] = patchShow(select_batch(P,channel,3), varargin{:});        
    end
    cLim = varargin{find(strcmp(varargin, 'CLim'))+1};
    I = I-cLim(1);
    I = I/(cLim(2)-cLim(1));
    I(I<0) = 0; I(I>1) = 1;
    if nnz(strcmp(varargin, 'noshow')) == 0
        imagesc(I);  
        set(gca,'XTick',meta.xTick);
        set(gca,'YTick',meta.yTick);        
        set(gca,'XTickLabel',meta.xTickLabels);        
        set(gca,'YTickLabel',meta.yTickLabels);
        set(gca,'ticklength',[0 0]);
        set(gca,'XAxisLocation','top')
    end
else

    if iscell(P)
        P = cat(ndims(P{1}) + 1, P{:});
    end
    if isnumeric(P)
        P = reshape(P, size(P,1), size(P,2), []);
        %if ndims(P) == 2 || (ndims(P) == 3 && size(P,3) == 1)
        %    error('Should have more than one image');
        %end
    end

    % handle parameters
    for i = 1:length(varargin)
        if isstr(varargin{i})
            varargin{i} = lower(varargin{i});
        end
    end

    findRows = find(strcmp(varargin, 'rows'));
    findCols = find(strcmp(varargin, 'cols'));
    findNumPatches = find(strcmp(varargin, 'numpatches'));
    findLineWidth = find(strcmp(varargin, 'linewidth'));
    findBgColor = find(strcmp(varargin, 'bgcolor'));
    findCLim = find(strcmp(varargin, 'clim'));
    findCLimPrc = find(strcmp(varargin, 'climprc'));

    if findRows, rows = varargin{findRows + 1}; end
    if findCols, cols = varargin{findCols + 1}; end
    if findNumPatches, N = varargin{findNumPatches + 1}; end
    if findLineWidth, lineWidth = varargin{findLineWidth + 1}; else, lineWidth = 1; end
    if findBgColor, bgColor = varargin{findBgColor + 1}; else, bgColor = 0; end
    if findCLim, cLim = varargin{findCLim + 1}; else, cLim = [min(P(:)) max(P(:))]; end;
    if findCLimPrc, cLimPrc = varargin{findCLimPrc + 1}; end

    % number of patches
    if ~exist('N', 'var') 
        if exist('rows', 'var') && exist('cols', 'var')
            N = rows * cols;
        else
            if ndims(P) >= 3
                N = size(P, ndims(P));
            else
                N = 1;
            end
        end
    end

    % patch size
    patchRows = size(P, 1);
    patchCols = size(P, 2);

    % number of rows and cols
    if ~exist('rows', 'var') && exist('cols', 'var')
        rows = ceil(N / cols);
    elseif ~exist('cols', 'var') && exist('rows', 'var')
        cols = ceil(N / rows);
    end

    if ~exist('rows', 'var') && ~exist('cols', 'var')
        rows = 1;
        cols = 1;

        while rows * cols < N
            cols = cols + 1;

            if (rows * cols < N) && (3 * cols * patchCols >  4 * rows * patchRows);
                rows = rows + 1;
            end
        end
    end

    if N > rows * cols;
        N = rows * cols;
    end

    % stick patches together
    if ndims(P) > 3
        bgColor = uint8(bgColor * 255);

        I = ones(...
              rows * patchRows + (rows + 1) * lineWidth, ...
              cols * patchCols + (cols + 1) * lineWidth, ...
              size(P, 3), 'uint8') * bgColor;
    else
        % normalize values
        %P = P(:, :, 1:min(N, end));
        %Pmin = min(P(:));
        %Pmax = max(P(:));
        %P = (P - Pmin) ./ (Pmax - Pmin);

        I = ones(...
              rows * patchRows + (rows + 1) * lineWidth, ...
              cols * patchCols + (cols + 1) * lineWidth) * bgColor;
    end

    for i = 0:(N - 1)
        c = mod(i, cols);
        r = floor(i/cols);
        rOff = (r + 1) * lineWidth + r * patchRows;
        cOff = (c + 1) * lineWidth + c * patchCols;

        if ndims(P) > 3
            I(rOff + 1:rOff + patchRows, cOff + 1:cOff + patchCols, :) = P(:, :, :, i + 1);
        else
            I(rOff + 1:rOff + patchRows, cOff + 1:cOff + patchCols) = P(:, :, i + 1);
        end
    end

    % show result
    scale=ceil(16/size(P,1));
    
    meta.xTick = [(patchCols+lineWidth)/2:patchCols+lineWidth:(patchCols+lineWidth)*cols];
    meta.yTick = [(patchRows+lineWidth)/2:patchRows+lineWidth:(patchRows+lineWidth)*rows];
    for n=1:cols
        xTickLabels{n} = num2str(n);
    end
    meta.xTickLabels = xTickLabels;
    for n=1:rows
        yTickLabels{n} = [num2str((n-1)*cols+1) '-' num2str(n*cols)];
    end
    meta.yTickLabels = yTickLabels;
    
    
    if nnz(strcmp(varargin, 'noshow')) == 0
        if findCLim
            %imagesc(I, (cLim - Pmin) ./ (Pmax - Pmin));
            imagesc(I, cLim);
        elseif findCLimPrc
            imagesc(I, prctile(P(:), cLimPrc));
        else
            imagesc(I);
        end


        %truesize;
        %axis equal off;

        if ndims(P) < 4
            colormap gray;
        end
        
        
        
        set(gca,'XTick',meta.xTick);
        set(gca,'YTick',meta.yTick);        
        set(gca,'XTickLabel',meta.xTickLabels);        
        set(gca,'YTickLabel',meta.yTickLabels);
        set(gca,'ticklength',[0 0]);
        set(gca,'XAxisLocation','top')
    
    end
    
end

end
