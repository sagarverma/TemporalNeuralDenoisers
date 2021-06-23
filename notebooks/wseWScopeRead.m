function [outFilePath] = wseWScopeRead(inFilePath, useCache)
%WSCWSCOPEREAD Processes a single *.wsc file into a *.mat file.
%   inFilePath: The path (either absolute or relative) of the *.wsc input file.
%   useCache(optional): Set this variable to false to force reprocessing.
%   outFilePath: The name of the *.mat output file (it is written in the
%   same directory.

    % Default value for useCache:
    if (nargin < 2)
        useCache = true;
    end
    % Tests if the input file exists:
    if(exist(inFilePath, 'file') ~= 2)
        error('The specfied input file %s doesn''t exist.', inFilePath);
    end
    % Checks if the extension is good:
    if(any(inFilePath((end-3):end) ~= '.wse'))
        warning('The extension %s of the input file is not the one expected');  
    end
    % The output file path:
    outFilePath = [inFilePath(1:(end-4)), '_wse.mat'];
    % For data caching:
    if (useCache && (exist(outFilePath, 'file') == 2))
        return;
    end

    % Load XML and get scope:
    root = xmlread(inFilePath);
    root = root.getDocumentElement;
    %scope = root.getElementsByTagName('scope');
    %if (root.getLength() < 1)
    %    outFile = fopen('test.mat', 'w');
    %    fclose(outFile);
    %    warning('No scope in WScope file');
    %elseif (scope.getLength() > 1)
    %    warning('More than one scope in WScope file');
    %end
    %scope = scope.item(0);
    
    % Variable info:
    traces = root.getElementsByTagName('traces');
    names = cell(1, traces.getLength());
    signed = true(1, traces.getLength());
    ops = cell(1, traces.getLength());
    for t = 1:traces.getLength()
        % Variable name:
        traceName = traces.item(t - 1).getElementsByTagName('Variable');
        name = '';
        if (traceName.getLength() > 0)
            traceName = traceName.item(0).getFirstChild();
            if (~isempty(traceName))
                if (traceName.getNodeType() == traceName.TEXT_NODE)
                    name = char(traceName.getTextContent());
                    toks = regexp(name, '^(.*) \((.*)-Var\)$', 'tokens');
                    if ~isempty(toks)
                        name = [toks{1}{2}, '_', toks{1}{1}];
                    else
                        while (name(1) == '_')
                            name(1) = [];
                        end
                        name = regexprep(name, ' \(.*\)', '');
                    end
                end
            end
        end
        names{1, t} = name;
        
        % Variable signed:
        traceSigned = traces.item(t - 1).getElementsByTagName('IsSigned');
        if (traceSigned.getLength() > 0)
            traceSigned = traceSigned.item(0).getFirstChild();
            signInfo = '';
            if (~isempty(traceSigned) && (traceSigned.getNodeType() == traceSigned.TEXT_NODE))
                signInfo = char(traceSigned.getTextContent());
            end
            switch (signInfo)
                case 'true'
                    signed(1, t) = true;
                case 'false'
                    signed(1, t) = false;
                otherwise
                    warning('Invalid sign information for trace %d', t);
            end
        else
            warning('No sign information for trace %d', t);
        end
        
        % Operation:
        traceOp = traces.item(t - 1).getElementsByTagName('Operation');
        if (traceOp.getLength() > 0)
            traceOp = traceOp.item(0).getFirstChild();
            if (~isempty(traceOp) && (traceOp.getNodeType() == traceOp.TEXT_NODE))
                ops{1, t} = char(traceOp.getTextContent());
            end
        else
            warning('No operation for trace %d', t);
        end
    end
    
    % Read data line by line:
    dataLines = root.getElementsByTagName('RowData');
    dataLinesNumber = dataLines.getLength();
    t = uint64(zeros(dataLines.getLength(), 1));
    data = int32(zeros(dataLines.getLength(), size(names, 2)));
    
    prog = waitbar(0, 'Parsing data (0%) ...', ...
                'CreateCancelBtn', 'setappdata(gcbf, ''canceling'', 1)');
    setappdata(prog, 'canceling', 0);
    for l = 1:dataLinesNumber
        if (getappdata(prog, 'canceling') == 1)
            break;
        end
        dataLine = dataLines.item(l - 1);
        time = dataLine.getElementsByTagName('time');
        if (time.getLength() ~= 1)
            warning('Data line %d has multiple time', l);
        end
        time = time.item(0).getFirstChild();
        if (time.getNodeType() == time.TEXT_NODE)
            t(l) = sscanf(char(time.getTextContent()), '%ld');
        else
            warning('Data line %d has invalid time', l);
        end
        dataColumns = dataLine.getElementsByTagName('data');
        if (dataColumns.getLength() < 1)
            warning('Data line %d is invalid', l);
            continue;
        elseif (dataColumns.getLength() > 1)
            warning('Data line %d has multiple data fields', l);
        end
        dataColumns = dataColumns.item(0).getElementsByTagName('*');
        if (dataColumns.getLength() ~= size(names, 2))
            warning('Data line %d has bad number of elements.', l);
        end
        for c = 1:dataColumns.getLength()
            dataColumn = dataColumns.item(c - 1).getFirstChild();
            if (dataColumn.getNodeType() == dataColumn.TEXT_NODE)
                data(l, c) = sscanf(char(dataColumn.getTextContent()), '%d');
            else
                warning('Data element at line %d, column %d is invalid', l, c);
            end
        end
        if (floor((l - 1)*100/dataLinesNumber) < floor(l*100/dataLinesNumber))
            waitbar(l/dataLinesNumber, prog, sprintf('Parsing data (%d%%) ...', round(l*100/dataLinesNumber)));
        end
    end
    delete(prog);
    
    % Saturate data to uint 16 values:
    if (any(any(data < 0)) || any(any(data > 65535)))
        warning('Invalid data extracted');
    end
    data(data < 0) = 0;
    data(data > 65535) = 65535;
    
    % Ensure time in seconds and increasing:
    [t, i] = sort(t, 'ascend');
    data = data(i, :);
    t = double(t - t(1))/10000000; % NOTE needed to change to 10 000 000 in wseWScopeRead
   
    % Insert nan when there are missing points:
    j = 0;
    for i = transpose(find(diff(t) > 0.003))
        t = [t(1:(i + j)); nan; t((i + j + 1):end)];
        data = [data(1:(i + j), :); zeros(1, size(data, 2)); data((i + j + 1):end, :)];
        j = j + 1;
    end
    
    % Extract data and manage signedness:
    for n = 1:size(names, 2)
        if (~isempty(names{n}))
            if (signed(n))
                traceData = data(:, n);
                traceData(traceData > 32767) = traceData(traceData > 32767) - 65536;
                eval(sprintf('%s = int16(traceData);', names{n}));
            else
                traceData = data(:, n);
                eval(sprintf('%s = uint16(traceData);', names{n}));
            end
            if (~isempty(ops{n}))
                traceDataScaled = evalOperation(traceData, ops{n}); %#ok Used below in eval
                eval(sprintf('%s_scaled = traceDataScaled;', names{n}));
            else
                eval(sprintf('%s_scaled = double(traceData);', names{n}));
            end
        end
    end
    
    names = names(~strcmp(names, ''));
    scaledNames = cellfun(@(name) sprintf('%s_scaled', name), names, 'UniformOutput', false);
    save(outFilePath, 't', names{:}, scaledNames{:});
end

function x = evalOperation(y, op)
    % Load Qs in memory:
    for i = 1:64
        eval(sprintf('Q%02d = %lu;', i, pow2(i) - 1));
    end
    % Eval operation:
    x = double(y);
    fprintf('Evaluating: "%s"\n', sprintf('x = %s;', op));
    eval(sprintf('x = %s;', op));
end