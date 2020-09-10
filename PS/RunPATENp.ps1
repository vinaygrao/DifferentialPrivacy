$teachersSet = @(250)
$datasets = @('mnist','svhn','cifar10')
$teacherMaxStepsSet = @(500,1000,2000,6000)
$lap_scaleSet = @(20,50)
$file = ""
$max_examples=@(100,1000)
# $teachersSet = @(10, 20, 30 , 50, 100)
# $datasets = @('mnist','svhn','cifar10') ,'svhn','cifar10'
# $teacherMaxStepsSet = @(500,1000,2000,6000)
# $lap_scaleSet = @(1,20,50)

$trainTeacherCmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate primary  & cd /d D:\Vinay\Repos\Differential%20Privacy\Resourses\privacy-masterv2\privacy-master\research\pate_2017  & python train_teachers.py  "
$trainStudentCmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate primary  & cd D:\Vinay\Repos\Differential%20Privacy\Resourses\privacy-masterv2\privacy-master\research\pate_2017  & python  train_student.py  "
$AnalysisCmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate primary  & cd D:\Vinay\Repos\Differential%20Privacy\Resourses\privacy-masterv2\privacy-master\research\pate_2017  & python  analysis.py  "

function runTeacher {
    Param($Rdataset,$Rteacher=20, $RteacherMaxSteps=3000, $Rlap_scale=20, $isRunAllTeacher, $teacherId)
    if($file -eq ""){
        if($Rdataset -eq "mnist"){ $file = "D:\Vinay\logs\PATE-PN\MNIST\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
        elseif($Rdataset -eq "svhn"){ $file = "D:\Vinay\logs\PATE-PN\SVHN\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
        elseif($Rdataset -eq "cifar10"){ $file = "D:\Vinay\logs\PATE-PN\cifar10\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
        else{ $file = "D:\Vinay\logs\PATE-PN\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"}

        $file = $file -replace '\s',''
    }    
    if($isRunAllTeacher){ 
                Write-Host("Treacher Training id: $($teacherId)") -ForegroundColor Green
                for ($i = 0; $i -lt $Rteacher; $i++) {              
                
                    $parameters = "--nb_teachers=$($Rteacher)  --teacher_id=$($i) --dataset=$($Rdataset)  --max_steps=$($RteacherMaxSteps)"
                    $commandDP = "$($trainTeacherCmd) $($parameters) "
                    Write-Host($commandDP) -ForegroundColor  White
                    $procDP = Start-Process cmd  "$($commandDP) >> $($file) " -wait             
                    
                }           
    
    }
    else {
        $parameters = "--nb_teachers=$($Rteacher)  --teacher_id=$($teacherId) --dataset=$($Rdataset)  --max_steps=$($RteacherMaxSteps)"
        $commandDP = "$($trainTeacherCmd) $($parameters) "
        Write-Host($commandDP) -ForegroundColor  White
        $procDP = Start-Process cmd  "$($commandDP) >> $($file) " -wait        
    }
    
  
       
}

function RunStudent {
    Param($Rdataset,$Rteacher=20, $RteacherMaxSteps=3000, $Rlap_scale=20,$RMaxExamples)
    if($file -eq ""){
    if($Rdataset -eq "mnist"){ $file = "D:\Vinay\logs\PATE-PN\MNIST\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    elseif($Rdataset -eq "svhn"){ $file = "D:\Vinay\logs\PATE-PN\SVHN\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    elseif($Rdataset -eq "cifar10"){ $file = "D:\Vinay\logs\PATE-PN\cifar10\Analysis\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    else{ $file = "D:\Vinay\logs\PATE-PN\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"}
    $file = $file -replace '\s',''
    }
    $parameters1 = "--nb_teachers=$($Rteacher)  --dataset=$($Rdataset) --lap_scale=$($Rlap_scale) --stdnt_share=5000  --teachers_max_steps=$($RteacherMaxSteps)"
    $commandDP1 = "$($trainStudentCmd) $($parameters1) "
    Write-Host($commandDP1) -ForegroundColor  Yellow
    $procDP1 = Start-Process cmd  "$($commandDP1) >> $($file) " -wait  

     
 }

function RunAnalysis {
    Param($Rdataset,$Rteacher=20, $RteacherMaxSteps=3000, $Rlap_scale=20,$RMaxExamples)

   
    if($Rdataset -eq "mnist"){ $file = "D:\Vinay\logs\PATE-PN\MNIST\Analysis\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    elseif($Rdataset -eq "svhn"){ $file = "D:\Vinay\logs\PATE-PN\SVHN\Analysis\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    elseif($Rdataset -eq "cifar10"){ $file = "D:\Vinay\logs\PATE-PN\cifar10\Analysis\Log_$($Rteacher)Teachers__$($RteacherMaxSteps)Maxsteps__$($Rlap_scale)lap__$($RMaxExamples)examples_$($(Get-Date -f 'dd-MM-yyyy-hh-mm-ss')).txt"}
    else{ $file = "D:\Vinay\logs\PATE-PN\Log_$($Rteacher)_Teachers-$($RteacherMaxSteps)_Maxsteps-$($Rlap_scale)_lap_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"}
    $file = $file -replace '\s',''
   

    $eps=1/$Rlap_scale
    #$parameters2 = "--input_is_counts --noise_eps=$($Rlap_scale/100) --counts_file=D:\Vinay\Data\$($Rdataset)_$($Rteacher)_teachers_labels_lap_$($Rlap_scale).npy --indices_file=D:\Vinay\Data\$($Rdataset)_$($Rteacher)_student_labels_lap_$($Rlap_scale).npy"
    $parameters2 = " --max_examples=$($RMaxExamples) --noise_eps=$($eps) --input_is_counts --counts_file=D:\Vinay\Data\$($Rdataset)_$($Rteacher)_teachers_labels_lap_$($Rlap_scale).npy "
    $commandDP2 = "$($AnalysisCmd) $($parameters2) "
    Write-Host($commandDP2)  -ForegroundColor Magenta
    $procDP2 = Start-Process cmd  "$($commandDP2) >> $($file) " -wait  
 }

function Teacher{
    foreach($dataset in $datasets)
    {
        # foreach($teacherMaxSteps in $teacherMaxStepsSet) 
        # {
           # runTeacher -Rdataset $dataset -RteacherMaxSteps  $teacherMaxSteps -isRunAllTeacher
        # }
        foreach($teacher in $teachersSet)
        {
            runTeacher -Rdataset $dataset -Rteacher  $teacher -teacherId 136 
        }
       
        # foreach($lap_scale in $lap_scaleSet)
        #  {
        #     runTeacher -Rdataset $dataset -Rlap_scale  $lap_scale -isRunAllTeacher
        # }
     
         
    }
}

function Student{
    foreach($dataset in $datasets)
    {
        # foreach($teacherMaxSteps in $teacherMaxStepsSet) 
        # {
        #     RunStudent -Rdataset $dataset -RteacherMaxSteps  $teacherMaxSteps
        # }
        foreach($teacher in $teachersSet)
        {
            RunStudent -Rdataset $dataset -Rteacher  $teacher
        }
       
        # foreach($lap_scale in $lap_scaleSet)
        #  {
        #     RunStudent -Rdataset $dataset -Rlap_scale  $lap_scale
        # }
     
         
    }
}


function Analysis {

    foreach($max_example in $max_examples){
        foreach($dataset in $datasets)
        {
            # foreach($teacherMaxSteps in $teacherMaxStepsSet) 
            # {
            #     RunAnalysis -Rdataset $dataset -RteacherMaxSteps  $teacherMaxSteps -RMaxExamples $max_example
            #     sleep -Seconds 5
            # }
            foreach($teacher in $teachersSet)
            {
                RunAnalysis -Rdataset $dataset -Rteacher  $teacher -RMaxExamples $max_example
                sleep -Seconds 5
            }
        
            # foreach($lap_scale in $lap_scaleSet)
            # {
            #     RunAnalysis -Rdataset $dataset -Rlap_scale  $lap_scale -RMaxExamples $max_example
            #     sleep -Seconds 5
            # }
        
            
        }
    }
    
    
}

#Teacher
Student
Analysis
