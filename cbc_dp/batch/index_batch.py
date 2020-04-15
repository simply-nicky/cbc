"""
index_batch.py - batching to Maxwell cluster script
"""
import configparser
import argparse
import subprocess
import os
import numpy as np
from datetime import datetime
from ..utils import OUT_PATH, chunkify, PROJECT_PATH

class JobConfig():
    """
    Class parser for ini files
    config_file - path to a config file
    geom_file - path to a experimental geometry config file
    """
    name_str = "{scan_num:03d}_{mode:s}_index"

    def __init__(self, mode, out_path, scan_num, pop_size, n_isl, gen_num, pos_tol, rb_tol, ang_tol):
        self.mode, self.scan_num, self.out_path = mode, scan_num, out_path
        self.pop_size, self.n_isl, self.gen_num = pop_size, n_isl, gen_num
        self.pos_tol, self.rb_tol, self.ang_tol = pos_tol, rb_tol, ang_tol
        self.name = self.name_str.format(scan_num=scan_num, mode=mode)
        self.param_dict = {'--scan_num': self.scan_num, '--pop_size': self.pop_size,
                           '--n_isl': self.n_isl, '--gen_num': self.gen_num,
                           '--pos_tol': self.pos_tol, '--rb_tol': self.rb_tol,
                           '--ang_tol': self.ang_tol}

    @classmethod
    def import_ini(cls, config_file):
        """
        Import from an ini file

        config_file - path to a file
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        mode = config.get('config', 'mode')
        out_path = config.get('config', 'out_path')
        scan_num = config.getint('config', 'scan_num')
        pop_size = config.getint('config', 'pop_size')
        n_isl = config.getint('config', 'n_islands')
        gen_num = config.getint('config', 'gen_number')
        pos_tol = np.array([float(tol) for tol in config.get('config', 'pos_tol').split()])
        rb_tol = config.getfloat('config', 'rb_tol')
        ang_tol = config.getfloat('config', 'ang_tol')
        return cls(mode=mode, out_path=out_path, scan_num=scan_num,
                   pop_size=pop_size, n_isl=n_isl, gen_num=gen_num,
                   pos_tol=pos_tol, rb_tol=rb_tol, ang_tol=ang_tol)

    def shell_parameters(self):
        """
        Return shell script parameters as a list of strings
        """
        params = [self.mode, self.out_path]
        for key in self.param_dict:
            if key == '--pos_tol':
                params.append(key)
                params.extend([str(tol) for tol in self.param_dict[key]])
            else:
                params.extend([key, str(self.param_dict[key])])
        return params

class JobBatcher():
    """
    sbatch job class to conduct index refinement

    config_file - path to a config ini file
    geom_file - path to an experimental geometry ini file
    """
    batch_cmd = "sbatch"
    frmt = '%m-%d-%y_%H-%M-%S'
    index_script = os.path.join(PROJECT_PATH, "index.sh")
    combine_script = os.path.join(PROJECT_PATH, "cbc_dp/batch/combine.sh")
    data_file = "{out_path:s}_{idx:03d}.h5"
    out_file = "{job_name:s}_{now:s}.out"
    err_file = "{job_name:s}_{now:s}.err"
    error_text = "Command '{cmd:s}' has returned an error (code {code:s}): {stderr:s}"
    job_size = 16

    def __init__(self, config_file, geom_file, rb_file):
        self.geom_file, self.rb_file = geom_file, rb_file
        self._init_pool(config_file)

    def _init_pool(self, config_file):
        config = JobConfig.import_ini(config_file)
        self.data_dir = os.path.join(OUT_PATH['scan'].format(config.scan_num), 'index')
        os.makedirs(self.data_dir, exist_ok=True)
        self.sbatch_dir = os.path.join(OUT_PATH['scan'].format(config.scan_num), 'sbatch_out')
        os.makedirs(self.sbatch_dir, exist_ok=True)
        self.out_filename = config.out_path
        self.pool = []
        for idx, n_isl in enumerate(chunkify(config.n_isl, self.job_size)):
            out_path = os.path.join(self.data_dir,
                                    self.data_file.format(out_path=self.out_filename, idx=idx))
            job = JobConfig(mode=config.mode, out_path=out_path, scan_num=config.scan_num,
                            pop_size=config.pop_size, n_isl=n_isl, gen_num=config.gen_num,
                            pos_tol=config.pos_tol, rb_tol=config.rb_tol, ang_tol=config.ang_tol)
            self.pool.append(job)

    @classmethod
    def now(cls):
        """
        Return current date and time string at the particular format
        """
        return datetime.now().strftime(cls.frmt)

    def sbatch_parameters(self, job_name):
        """
        Return sbatch command parameters for a job
        """
        sbatch_params = ['--partition', 'upex', '--job_name', job_name,
                         '--output', os.path.join(self.sbatch_dir,
                                                  self.out_file.format(job_name=job_name, now=self.now())),
                         '--error', os.path.join(self.sbatch_dir,
                                                 self.err_file.format(job_name=job_name, now=self.now()))]
        return sbatch_params

    def index_command(self, job):
        """
        Return a command to batch an indexing job
        """
        command = [self.batch_cmd]
        command.extend(self.sbatch_parameters(job.name))
        command.extend([self.index_script, self.geom_file, self.rb_file])
        command.extend(job.shell_parameters())
        return command

    def combine_command(self, job_nums):
        """
        Return a command to batch a combine job
        """
        command = [self.batch_cmd]
        command.extend(self.sbatch_parameters('combine'))
        command.extend(['--dependency', 'afterok:{:s}'.format(':'.join(job_nums)), self.combine_script])
        command.extend([job.out_path for job in self.pool])
        command.append(self.out_filename + '.h5')
        return command

    def batch_job(self, job_name, command, test):
        """
        Batch a job
        """
        print('Submitting job: {:s}'.format(job_name))
        print('Command: {:s}'.format(' '.join(command)))
        if test:
            return '-1'
        else:
            try:
                output = subprocess.run(args=command, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                err_text = self.error_text.format(cmd=' '.join(command),
                                                  code=error.returncode,
                                                  stderr=error.stderr)
                raise RuntimeError(err_text) from error
            job_num = output.stdout.rstrip().decode("unicode_escape").split()[-1]
            print("The job {} has been submitted".format(job_name))
            print("Job ID: {}".format(job_num))
            return job_num

    def batch(self, test=False):
        """
        Batch a pool of jobs
        """
        job_nums = []
        for job in self.pool:
            command = self.index_command(job)
            job_nums.append(self.batch_job(job.name, command, test))
        command = self.combine_command(job_nums)
        self.batch_job('combine', command, test)

def main():
    parser = argparse.ArgumentParser(description='Batch to Maxwell jobs of indexing refinement')
    parser.add_argument('config_file', type=str, help='Path to a config ini file')
    parser.add_argument('geom_file', type=str, help='Path to an experimental geometry ini file')
    parser.add_argument('rb_file', type=str, help='Path to a reciprocal lattice basis vectors ini file')
    parser.add_argument('--test', action='store_true', help='Test batching the job to the Maxwell cluster')
    args = parser.parse_args()

    batcher = JobBatcher(args.config_file, args.geom_file, args.rb_file)
    batcher.batch(test=args.test)
