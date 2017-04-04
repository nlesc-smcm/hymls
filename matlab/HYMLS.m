classdef HYMLS < handle
    properties
        preconditioner
    end
    methods
        function h = HYMLS(A, params)
            if nargin ~= 2
                error('Two input arguments required');
            end
            h.preconditioner = HYMLS_init(A, params);
        end

        function y = apply(h, x)
            if nargin ~= 2
                error('One input argument required');
            end
            y = HYMLS_apply(h.preconditioner, x);
        end

        function delete(h)
            if ~isempty(h.preconditioner)
                HYMLS_free(h.preconditioner);
                h.preconditioner = [];
                fprintf('HYMLS successfully deleted\n');
            end
        end
    end
end