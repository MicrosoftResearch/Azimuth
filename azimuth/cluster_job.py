from IPython.parallel.apps.winhpcjob import *
import tempfile
import pickle

# just execute this file in python to create the xml file for the cluster (in ./analysis/cluster), which one then can manually submit through the HPC Job Manager

def cluster_setup(i, python_path, home, t, work_dir, tempdir):
    t.work_directory = work_dir    
    #t.std_out_file_path = r'cluster\log\cluster_out%d.txt' % i
    #t.std_err_file_path = r'cluster\log\cluster_err%d.txt' % i
    t.std_out_file_path = tempdir + r'\out%d.txt' % i
    t.std_err_file_path = tempdir + r'\err%d.txt' % i
    #t.std_out_file_path = r'out%d.txt' % i
    #t.std_err_file_path = r'err%d.txt' % i
    #if not os.path.exists(t.std_out_file_path): os.makedirs(t.std_out_file_path)
    #if not os.path.exists(t.std_err_file_path): os.makedirs(t.std_err_file_path)
    t.environment_variables['PYTHONPATH'] = python_path     
    t.environment_variables['HOME'] = home

    print "cluster python_path=%s" % python_path

def create(user, models, orders, degrees, GP_likelihoods, adaboost_learning_rates=None, adaboost_num_estimators=None, adaboost_max_depths=None, adaboost_CV=False, exp_name=None, learn_options=None):
    job = WinHPCJob()
    job.job_name = 'CRISPR.%s' % exp_name
    if user == 'fusi':
        job.username = 'REDMOND\\fusi'
    elif user =='jennl':
        job.username = 'REDMOND\\jennl'
    else:
        raise Exception("ensure you are using the right username, then add a clause here")
    job.priority = 'Normal'
    job.min_nodes = 1
    job.max_nodes = 5000
    job.min_cores = 1
    job.max_cores = 5000

    if job.username == 'REDMOND\\fusi':
        remote_dir = r'\\fusi1\crispr2\analysis\cluster\results'
        work_dir = r"\\FUSI1\crispr2\analysis" #working dir wherever you put it, even off the cluster
        python = r'\\fusi1\crispr\python.exe'  #this will not get copied, but used in-place
        python_path = r'\\fusi1\crispr\lib\site-packages\;\\fusi1\crispr2\analysis'
        home =  r"\\fusi1\CLUSTER_HOME"
    elif job.username == 'REDMOND\\jennl':
        remote_dir = r"\\GCR\Scratch\RR1\jennl\CRISPR"
        work_dir = r'\\jennl2\D$\Source\CRISPR\analysis'            
        python = r'\\fusi1\crispr\python.exe'
        python_path = r'\\fusi1\crispr\lib\site-packages\;\\jennl2\D$\Source\CRISPR\analysis'
        home =  r"\\fusi1\CLUSTER_HOME"

    # print "workdir=%s" % work_dir
    # print "python=%s" % python
    # print "python_path=%s" % python_path

    # generate random dir in results directory   
    tempdir = tempfile.mkdtemp(prefix='cluster_experiment_', dir=remote_dir)
    print "Created directory: %s" % str(tempdir)

    # dump learn_options
    with open(tempdir+'/learn_options.pickle', 'wb') as f:
        pickle.dump(learn_options, f)

    i = 0
    for model in models:
        if model in ['L1', 'L2', 'linreg', 'doench', 'logregL1', 'RandomForest', 'SVC']:
            for order in orders:
                t = WinHPCTask()
                t.task_name = 'CRISPR_task'
                t.command_line = python + ' cli_run_model.py %s --order %d --output-dir %s --exp-name %s' % (model, order, tempdir, exp_name)
                cluster_setup(i, python_path, home, t, work_dir, tempdir)
                t.min_nodes = 1
                # t.min_cores = 1
                # t.max_cores = 100
                job.add_task(t)
                i += 1

        elif model in ['AdaBoost']:
            for order in orders:
                for learning_rate in adaboost_learning_rates:
                    for num_estimators in adaboost_num_estimators:
                        for max_depth in adaboost_max_depths:
                            t = WinHPCTask()
                            t.task_name = 'CRISPR_task'
                            t.command_line = python + ' cli_run_model.py %s --order %d --output-dir %s --adaboost-learning-rate %f --adaboost-num-estimators %d --adaboost-max-depth %d --exp-name %s' % (model, order, tempdir, learning_rate, num_estimators, max_depth, exp_name)
                            if adaboost_CV:
                                t.command_line += " --adaboost-CV"
                            cluster_setup(i, python_path, home, t, work_dir, tempdir)
                            t.min_nodes = 1
                            job.add_task(t)
                            i += 1

        elif model in ['GP']:
            for likelihood in GP_likelihoods:
                for degree in degrees:
                    t = WinHPCTask()
                    t.task_name = 'CRISPR_task'
                    t.command_line = python + ' cli_run_model.py %s --order 1 --weighted-degree %s --output-dir %s --likelihood %s --exp-name %s' % (model, degree, tempdir, likelihood, exp_name)
                    cluster_setup(i, python_path, home, t, work_dir, tempdir)
                    t.min_nodes = 1
                    # t.min_cores = 1
                    # t.max_cores = 100
                    job.add_task(t)
                    i += 1

        else:
            t = WinHPCTask()
            t.task_name = 'CRISPR_task'
            t.command_line = python + ' cli_run_model.py %s --output-dir %s' % (model, tempdir)
            cluster_setup(i, python_path, home, t, work_dir, tempdir)
            t.min_cores = 1
            t.max_cores = 1

            job.add_task(t)
            i += 1


    clust_filename = 'cluster_job.xml'
    job.write(tempdir+'/'+clust_filename)
    return tempdir, job.username, clust_filename
