classdef test_stokes0 < matlab.unittest.TestCase
    properties
    end

    methods (Test)
        function testStokes0(testCase)
            A = mmread([fileparts(mfilename('fullpath')) ...
                        '/../../testSuite/data/DrivenCavity/32x32/Re0/jac.mtx']);
            P = HYMLS_init(A, [fileparts(mfilename('fullpath')) ...
                               '/../../testSuite/integration_tests/stokes0.xml']);
            b = mmread([fileparts(mfilename('fullpath')) ...
                        '/../../testSuite/data/DrivenCavity/32x32/Re0/rhs.mtx']);
            x = HYMLS_apply(P, b);

            % Add border for pressure part
            n = size(b, 1);
            idxp = 3:3:n;
            v = zeros(n, 1);
            v(idxp) = 1;
            A2 = [A, v; v', zeros(size(v,2))];
            b2 = [b; zeros(size(v,2), 1)];
            x2 = A2 \ b2;
            x2 = x2(1:n, :);

            % Remove pressure part
            idxv = 1:n;
            idxv(idxp) = [];
            testCase.verifyLessThan(norm(x(idxv)-x2(idxv)), 1e-9);

            % Use Dirichlet boundary condition
            A(3,:) = sparse(1, n);
            A(3,3) = 1;
            x3 = A \ b;
            testCase.verifyLessThan(norm(x-x3), 1e-9);
        end
    end
end