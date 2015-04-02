import subprocess
import threading
import os
from optparse import OptionParser
import sys
import datetime
import shutil

def print_input(nodes, procs, size, levels, ssize, re_start, re_end, re_step, restart):
    text = '''<ParameterList name="Driven Cavity using LOCA/HYMLS"><!--{-->

  <ParameterList name="Model"><!--{-->
  
     <Parameter name="dim" type="int" value="3"/>
    <Parameter name="dof" type="int" value="4"/>
     
     <!-- number of interior cells -->
    <Parameter name="nx" type="int" value="%d"/>
    <Parameter name="ny" type="int" value="%d"/>
    <Parameter name="nz" type="int" value="%d"/>

    <Parameter name="xmin" type="double" value="0.0"/>
    <Parameter name="xmax" type="double" value="1.0"/>
    <Parameter name="ymin" type="double" value="0.0"/>
    <Parameter name="ymax" type="double" value="1.0"/>
    <Parameter name="zmin" type="double" value="0.0"/>
    <Parameter name="zmax" type="double" value="1.0"/>
    
    <Parameter name="cell ratio (x)" type="double" value="1.0"/>
    <Parameter name="cell ratio (y)" type="double" value="1.0"/>
    <Parameter name="cell ratio (z)" type="double" value="1.0"/>
    
    <Parameter name="Dump all Linear Systems" type="bool" value="0"/>
    <!-- this flag enables adapting the eigenvalue strategy after -->
    <!-- a continuation step (experimental)                       -->
    <Parameter name="Adaptive Cayley" type="bool" value="false"/>


    <Parameter name="Output Frequency" type="double" value="-1.0"/>
    <Parameter name="Backup Interval" type="double" value="-1.0"/>

    <!-- restarts from the data specified in the XML file 'Restart File' -->
    <!-- set to "None" to start from zero solution and default params    -->
    <!-- the standard name for a restart file of cavity is restart.xml   -->
    <Parameter name="Restart File" type="string" value="%s"/>
   
    <!--{--> 
    <ParameterList name="Starting Parameters">    
      <Parameter name="Reynolds Number" type="double" value="%d.0"/>
      <Parameter name="Coefficient Number" type="double" value="0.0"/>
    </ParameterList><!--}-->
  </ParameterList><!--}-->
  
  <!-- LOCA parameters - anything that concerns the continuation process { -->
  <ParameterList name="LOCA">

    <!-- We do not do bifurcation analysis yet, but here is the place -->
    <!-- the parameters for it:                                     { -->
    <ParameterList name="Bifurcation">
      <Parameter name="Method" type="string" value="None"/>
    </ParameterList> <!-- } Bifurcation -->
    
    <!-- Parameters for the predictor method {-->
    <ParameterList name="Predictor">
      <!-- choices are "Constant","Tangent" and "Secant" here -->
      <Parameter name="Method" type="string" value="Constant"/>
      <ParameterList name="First Step Predictor"> <!-- { -->
        <Parameter name="Method" type="string" value="Constant"/>
      </ParameterList> <!-- } First Step Predictor -->
    </ParameterList> <!-- } Predictor -->
    
    <!-- Parameters influencing step size control in the continuation { -->
    <ParameterList name="Step Size">

      <!-- possible choices are "Constant" or "Adaptive" -->
      <Parameter name="Method" type="string" value="Adaptive"/>

      <Parameter name="Initial Step Size" type="double" value="%d.0"/>
      <Parameter name="Min Step Size" type="double" value="1.0"/>
      <Parameter name="Max Step Size" type="double" value="%d.0"/>

      <Parameter name="Failed Step Reduction Factor" type="double" value="0.5"/>

      <!-- This affects the "Adaptive" mode only -->
      <Parameter name="Aggressiveness" type="double" value="0.5"/>

      <!-- This affects the "Constant" mode only -->
      <Parameter name="Successful Step Increase Factor" type="double" value="1.95"/>
    </ParameterList><!-- } Step Size -->
    
    <!-- Parameters for the parameter advancing during the continuation { -->
    <ParameterList name="Stepper">
      <!-- "Natural" or "Arc Length" -->
      <Parameter name="Continuation Method" type="string" value="Natural"/>
      <!-- currently supported: "Reynolds Number" -->
      <Parameter name="Continuation Parameter" type="string" value="Reynolds Number"/>
      <Parameter name="Initial Value" type="double" value="0.0"/>
      <Parameter name="Min Value" type="double" value="%d.0"/>
      <Parameter name="Max Value" type="double" value="%d.0"/>
      <Parameter name="Max Steps" type="int" value="250"/>
      <!-- I think this is irrelevant, use "Line Search"->"Max Iters" instead -->
      <Parameter name="Max Nonlinear Iterations" type="int" value="10"/>
      <Parameter name="Compute Eigenvalues" type="bool" value="0"/>
      <!-- these settings have _not_ been tested at all { -->
      <ParameterList name="Eigensolver">
        <!-- "Anasazi" (Arnoldi) or "Default" (means no eigensolver) -->
        <Parameter name="Method" type="string" value="Anasazi"/>
        <!-- "Shift-Invert": find eigs near Cayley pole             -->
        <!-- We use "Cayley" to compute eienvalues                 -->
        <!-- (A-sigma*B)\\(A-mu*B)                                   -->
        <Parameter name="Operator" type="string" value="Cayley"/>
        <!-- sigma if you choose the Shift-Invert operator -->
        <Parameter name="Shift" type="double" value="0.0"/>
        <!-- sigma if you choose the Cayley operator -->
        <Parameter name="Cayley Pole" type="double" value="1.0"/>
        <!-- mu if you choose the Cayley operator -->
        <Parameter name="Cayley Zero" type="double" value="-3.5"/>
        <!-- Order of printed eigenvalues -->
        <Parameter name="Sorting Order" type="string" value="LM"/>
        
        <!-- settings for the outer Krylov iteration -->
        
        <!-- block size for Block Krylov method -->
        <Parameter name="Block Size" type="int" value="1"/>
        <Parameter name="Convergence Tolerance" type="double" value="2e-5"/>
        <Parameter name="Maximum Restarts" type="int" value="3"/>
        <!-- length of Krylov sequence -->
        <Parameter name="Num Blocks" type="int" value="350"/>
        <Parameter name="Num Eigenvalues" type="int" value="25"/>
        <!-- convergence check after this many steps -->
        <Parameter name="Step Size" type="int" value="5"/>
        <Parameter name="Linear Solve Tolerance" type="double" value="1.0e-6"/>
      </ParameterList> <!-- } Eigensolver -->
      
      <!-- "Bordering" (default), "Householder" (prec reuse not working) -->
      <Parameter name="Bordered Solver Method" type="string" value="Bordering"/>
      <Parameter name="Enable Arc Length Scaling" type="bool" value="0"/>
      <Parameter name="Initial Scale Factor" type="double" value="1"/>
      <Parameter name="Min Scale Factor" type="double" value="1e-08"/>
      <Parameter name="Goal Arc Length Parameter Contribution" type="double" value="0.5"/>
      <Parameter name="Max Arc Length Parameter Contribution" type="double" value="0.7"/>
      <Parameter name="Enable Tangent Factor Step Size Scaling" type="bool" value="0"/>
      <Parameter name="Min Tangent Factor" type="double" value="0.1"/>
      <Parameter name="Tangent Factor Exponent" type="double" value="1"/>
    </ParameterList> <!-- } Stepper -->
  </ParameterList><!-- } LOCA -->
    
    
  <!-- Nonlinear solver parameters { -->
  <ParameterList name="NOX">

    <!-- "Line Search Based" or "Trust Region Based" (not tested) -->
    <Parameter name="Nonlinear Solver" type="string" value="Line Search Based"/>
    <!-- convergence tolerance for the Newton process -->
    <Parameter name="Convergence Tolerance" type="double" value="1.0e-9"/>

    <!-- Line Search parameters { -->
    <ParameterList name="Line Search">
      <!-- "Full Step" is standard Newton's, "Backtrack" is a line search, -->
      <!-- others exist as well -->
      <Parameter name="Method" type="string" value="Full Step"/>
      
      <!-- maximum number of Newton iterations -->
      <Parameter name="Max Iters" type="int" value="10"/>
      
      <ParameterList name="Backtrack">
        <Parameter name="Default Step" type="double" value="1.0"/>
        <Parameter name="Max Iters" type="int" value="10"/>
        <Parameter name="Minimum Step" type="double" value="1e-6"/>
        <Parameter name="Recovery Step" type="double" value="1e-3"/>
      </ParameterList>
    </ParameterList><!-- } Line Search -->
    
    <!-- Parameters for the actual Newton solver { -->
    <ParameterList name="Direction">
      <Parameter name="Method" type="string" value="Newton"/>
      
      <!-- { -->
      <ParameterList name="Newton">
        
        <!-- Method to determine convergence tolerance for linear solver -->
        <!-- "Constant", "Type 1" (fails!) or "Type 2" -->
        <Parameter name="Forcing Term Method" type="string" value="Constant"/>
        <Parameter name="Forcing Term Initial Tolerance" type="double" value="1.0e-3"/>
        <Parameter name="Forcing Term Maximum Tolerance" type="double" value="1.0e-2"/>
        <Parameter name="Forcing Term Minimum Tolerance" type="double" value="1.0e-8"/>
        
        <!-- Use Newton step even if linear solve failed (default 1)-->
        <!-- If you say "0" here the continuation run _stops_ when  -->
        <!-- the linear solver fails to achieve the requested tol   -->
        <Parameter name="Rescue Bad Newton Solve" type="bool" value="1"/>
        
        <!-- Note: the docu for this list is in class NOX::Epetra::LinearSystemAztecOO { -->
        <ParameterList name="Linear Solver">
        
          <Parameter name="Tolerance" type="double" value="1.0e-8"/>

          <Parameter name="Write Linear System" type="bool" value="0"/>

          <!-- choose a preconditioner for the linear solves during Newton steps        -->
          <!-- "User Defined": hymls (see 'HYMLS' sublist)                              -->
          <!-- "New Ifpack": incomplete factorizations, parameters are set below        -->
          <!-- "ML": algebraic multigrid, parameters are set below                      -->
          <Parameter name="Preconditioner" type="string" value="User Defined"/>

          <!-- can be "Reuse", "Rebuild" or "Recompute"        -->
          
          <!-- "Rebuild" and "Recompute" are the same for most methods.           -->
          <!-- "Reuse" keeps the preconditioner for k linear system solves.       -->
          <!-- If you want to reuse the preconditioner at all, set this to "Reuse"-->
          <Parameter name="Preconditioner Reuse Policy" type="string" value="Rebuild"/>
          
          <!-- recompute preconditioner after K linear system solves. -->
          
          <!-- We have our own reuse-policy, controlled by the option -->
          <!-- with the same name in thcm_params.xml. If you want to  -->
          <!-- reuse the preconditioner, it's best to set this to -2  -->
          <!-- and use that option to control the maximum age.        -->
          <Parameter name="Max Age Of Prec" type="int" value="10"/>
       
          <!-- linear solver parameters                                         -->
          <!-- the solver has a number of options that are set in class         -->
          <!-- DrivenCavity and you shouldn't have to worry about them.         -->
          <ParameterList name="HYMLS"><!--{-->

            <!-- { -->
            <ParameterList name="Solver">
            
              <!-- use deflation to handle the null space -->
              <!--Parameter name="Null Space" type="string" value="Constant P"/-->
              <Parameter name="Deflated Subspace Dimension" type="int" value="0"/>

                      <!-- "CG", "GMRES", "PCPG" (only GMRES actually supported by our 
               solver right now) -->
              <Parameter name="Krylov Method" type="string" value="GMRES"/>
              
              <!-- start vector to use for the Krylov sequence ("Zero","Previous","Random")-->
              <Parameter name="Initial Vector" type="string" value="Zero"/>

              <!-- parameters for the iterative solver (Belos) { -->
              <ParameterList name="Iterative Solver">
                <Parameter name="Maximum Iterations" type="int" value="10000"/>
                <Parameter name="Num Blocks" type="int" value="200"/>
                <Parameter name="Maximum Restarts" type="int" value="200"/>
                <Parameter name="Flexible Gmres" type="bool" value="0"/>
                <Parameter name="Convergence Tolerance" type="double" value="1.0e-8"/>
                <Parameter name="Output Frequency" type="int" value="1"/>
                <Parameter name="Show Maximum Residual Norm Only" type="bool" value="1"/>
              </ParameterList><!--}-->
            </ParameterList><!--}-->

            <!-- { -->
            <ParameterList name="Preconditioner">
              <!-- choose basic partitioning method. The only choice -->
              <!-- available right now is 'Cartesian'                -->
              <Parameter name="Partitioner" type="string" value="Cartesian"/>
              <!-- this defines how separator nodes are classified. -->
              <!-- the default is 'Standard', which gives full con- -->
              <!-- servation cells all along the edges of the 3D    -->
              <!-- subdomains. A 3D Stokes specific variant is also -->
              <!-- available as 'Stokes', which eliminates the in-  -->
              <!-- terior of the edges (full conservation tubes)    -->
              <!-- and is therefore much faster. We can also use'Hybrid'                   -->
              <Parameter name="Classifier" type="string" value="Hybrid"/>
              
              <!-- this means that a single P-row is replaced by a -->
              <!-- Dirichlet condition in the last SC. This should -->
              <!-- be done unless you use bordering to fix the     -->
              <!-- singularity (cf. 'Solver'->'Null Space' option) -->
              <Parameter name="Fix Pressure Level" type="bool" value="true"/>

              <!-- write file hid_data.m -->
              <Parameter name="Visualize Solver" type="bool" value="false"/>
                
              <!-- this defines the size of the subdomains -->
              <Parameter name="Separator Length" type="int" value="%d"/>
              <Parameter name="Base Separator Length" type="int" value="2"/>

              <!-- number of levels to be created.                      -->
              <!-- 1: direct solver for Schur complement,               -->
              <!-- 2: compute SC, transform+drop, solve directly.       -->
              <!-- 3: recursive application                             -->
              <Parameter name="Number of Levels" type="int" value="%d"/>

              <!-- switch to dense solver on next-to-last level -->
              <Parameter name="Subdomain Solver Type" type="string" value="Sparse"/>

              <!-- scale those rows of the Schur complement that are not coupled to -->
              <!-- P-nodes so that they have ones on the diagonal                   -->
              <Parameter name="Scale Schur-Complement" type="bool" value="0"/>

              <ParameterList name="Sparse Solver"><!--{-->
                <Parameter name="amesos: solver type" type="string" value="KLU"/>
                <Parameter name="Custom Ordering" type="bool" value="1"/>
                <Parameter name="Custom Scaling" type="bool" value="1"/>
                <Parameter name="OutputLevel" type="int" value="0"/>
              </ParameterList><!--}-->

              <ParameterList name="Dense Solver"><!--{-->
              </ParameterList><!--}-->

              <ParameterList name="Coarse Solver"><!--{-->
                <Parameter name="amesos: solver type" type="string" 
                        value="Amesos_Lapack"/>
                <Parameter name="OutputLevel" type="int" value="0"/>
                <Parameter name="PrintTiming" type="bool" value="1"/>
                <Parameter name="PrintStatus" type="bool" value="0"/>
                <Parameter name="Redistribute" type="bool" value="1"/>
                <Parameter name="MaxProcs" type="int" value="16"/>
                <ParameterList name="Superludist">
                  <Parameter name="PrintNonzeros" type="bool" value="1"/>
                  <Parameter name="IterRefine" type="string" value="NONE"/>
                </ParameterList>
              </ParameterList><!--}-->
            </ParameterList><!--}-->
          </ParameterList><!--}-->          
          
          <!-- Ifpack Preconditioner for complete system: --> 

          <!-- can be "ILU","ILUT","Amesos" and others    -->
          <!-- Only relevant if you chose "Ifpack" above. -->
          <Parameter name="Ifpack Preconditioner" type="string" 
                value="Amesos stand-alone"/>

          <!-- used by Ifpack Additive Schwarz preconditioners: -->
          <Parameter name="Overlap" type="int" value="0"/>

          <ParameterList name="Ifpack"><!--{-->

            <!-- if you chose Amesos above, select the direct solver here:  -->
            <!-- "Amesos_Klu" is always available                           -->
            <!-- "Amesos_Mumps" is available on Aster at least              -->
            <Parameter name="amesos: solver type" type="string" 
            value="Amesos_Klu"/>
              
            <!-- general Amesos settings -->
            <Parameter name="OutputLevel" type="int" value="2"/>
            <Parameter name="PrintTiming" type="bool" value="1"/>
            <Parameter name="PrintStatus" type="bool" value="1"/>
            <Parameter name="NoDestroy" type="bool" value="0"/>

            <ParameterList name="MRILU"><!--{-->
              <Parameter name="blocksize" type="int" value="3"/>
              <Parameter name="cutmck" type="int" value="0"/>
              <Parameter name="scarow" type="int" value="1"/>
              <Parameter name="xactelm" type="int" value="1"/>
              <Parameter name="clsonce" type="int" value="0"/>
              <Parameter name="nlsfctr" type="double" value="0.1"/>
              <Parameter name="epsw" type="double" value="0.1"/>
              <Parameter name="elmfctr" type="double" value="0.2"/>
              <Parameter name="gusmod" type="int" value="1"/>
              <Parameter name="gusfctr" type="double" value="0.95"/>
              <Parameter name="redfctr" type="double" value="2.0"/>
              <Parameter name="schtol" type="double" value="1.0e-6"/>
              <Parameter name="denslim" type="double" value="0.9"/>
              <Parameter name="globfrac" type="double" value="0.05"/>
              <Parameter name="locfrac" type="double" value="0.0"/>
              <Parameter name="sparslim" type="double" value="0.95"/>
              <Parameter name="ilutype" type="int" value="9"/>
              <Parameter name="droptol" type="double" value="1.0e-5"/>
              <Parameter name="compfct" type="double" value="1.0"/>
              <Parameter name="cpivtol" type="double" value="0.875"/>
              <Parameter name="lutol" type="double" value="1.0e-10"/>
              <Parameter name="singlu" type="int" value="1"/>
              <Parameter name="Output Level" type="int" value="0"/> 

              <ParameterList name="visualization">
                <Parameter name="visasc" type="int" value="0"/>
                <Parameter name="visnsc" type="int" value="0"/>
                <Parameter name="vislsc" type="int" value="0"/>
                <Parameter name="visildu" type="int" value="0"/> 
                <Parameter name="visnro" type="int" value="0"/>
              </ParameterList> 
            </ParameterList><!-- } end of MRILU -->
                
            <!-- this is how to set MUMPS parameters in Amesos. See the  -->
            <!-- MUMPS user's guide for details                          -->
            <!-- In general the default settings made by Amesos are ok   -->
            <ParameterList name="mumps">
              <!-- error stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(1)" type="int" value="6"/>
              <!-- info stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(2)" type="int" value="6"/>
              <!-- global info stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(3)" type="int" value="6"/>
              <!-- verbosity (-1..3) -->
              <Parameter name="ICNTL(4)" type="int" value="3"/>
              <!-- collect statistics (for optimal performance set it to 0!) -->
              <Parameter name="ICNTL(11)" type="int" value="1"/>               
              <!-- this controls how much extra memory the solver may use, the -->
              <!-- default (20%%) is typically too small for our application    -->
              <Parameter name="ICNTL(14)" type="int" value="80"/>
              <!--Parameter name="CNTL(1)" type="double" value="?"/-->
            </ParameterList>
          </ParameterList><!-- } global Ifpack -->
        </ParameterList><!-- } Linear Solver -->
      </ParameterList><!-- } Newton -->
    </ParameterList><!-- } direction -->    
  </ParameterList><!-- } NOX -->
</ParameterList><!--}-->''' % (size, size, size, 'None' if not restart else restart, re_start, re_step, re_step, re_start, re_end, ssize, levels)

    f = open('FVM_LDCav_nn%02d_np%d_nx%d_sx%d_L%d_%d.xml' % (nodes, procs, size, ssize, levels, re_end), 'w')
    f.write(text)
    f.close()

    f = open('params.xml', 'w')
    f.write(text)
    f.close()

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            print('Thread started')
            self.process = subprocess.Popen(self.cmd, shell=True    , executable="/bin/bash")
            self.process.communicate()
            print('Thread finished at ' + str(datetime.datetime.now()))

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        killed = False
        if thread.is_alive():
            print('Terminating process')
            subprocess.call('killall -9 '+self.cmd.partition(' ')[0], shell=True)
            subprocess.call('killall -9 '+self.cmd.split(' ')[10], shell=True)
            thread.join()
            killed = True

        if self.process is None:
            return (-1, killed)

        print('Returncode is ' + str(self.process.returncode))
        return (self.process.returncode, killed)

def test_method(nodes, procs, size, levels, ssize, re_start, re_end, re_step, restart):
    fname = 'FVM_LDCav_nn%02d_np%d_nx%d_sx%d_L%d_%d' % (nodes, procs, size, ssize, levels, re_end)

    if restart:
        print_input(nodes, procs, size, levels, ssize, re_start, re_end, re_step, 'restart.xml')
    else:
        print_input(nodes, procs, size, levels, ssize, re_start, re_end, re_step, False)

    exe = 'srun --ntasks-per-node=%d env OMP_NUM_THREADS=1 /home/baars/stable/fredwubs/fvm/src/LDCavCont %s.xml &> %s.out' % (procs // nodes, fname, fname)

    print exe

    c = Command(exe)
    ret, killed = c.run(10000)

    #~ if os.path.isfile('restart.xml'):
        #~ shutil.copy('restart.xml', fname + '.restart')

if __name__ == '__main__':
    parser = OptionParser(usage="python runtest.py [options] dir_name") 
    parser.add_option("-t", "--test", action="store_true", dest="test", default=False,
                      help="give information about what is going to be checked without running")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                      help="give information about what is going to be checked")

    (options, args) = parser.parse_args()

    if len(args) < 5:
        print 'Invalid parameters'
        sys.exit()

    test_method(int(args[0]), int(args[1]), int(args[2]), int(args[4]), int(args[3]), 0,
    0, 500, False)
