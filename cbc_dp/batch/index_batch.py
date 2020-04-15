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
    shell_script = os.path.join(PROJECT_PATH, "index.sh")
    output_path = "index/{out_path:s}_{idx:03d}.h5"
    output_file = "sbatch_out/{job_name:s}_{now:s}.out"
    error_file = "sbatch_out/{job_name:s}_{now:s}.err"
    error_text = "Command '{cmd:s}' has returned an error (code {code:s}): {stderr:s}"
    job_size = 16

    def __init__(self, config_file, geom_file, rb_file):
        self.geom_file, self.rb_file = geom_file, rb_file
        self._init_pool(config_file)

    def _init_pool(self, config_file):
        config = JobConfig.import_ini(config_file)
        self.pool = []
        for idx, n_isl in enumerate(chunkify(config.n_isl, self.job_size)):
            out_path = os.path.join(OUT_PATH['scan'].format(config.scan_num),
                                    self.output_path.format(out_path=config.out_path, idx=idx))
            job = JobConfig(mode=config.mode, out_path=out_path, scan_num=config.scan_num,
                            pop_size=config.pop_size, n_isl=n_isl, gen_num=config.gen_num,
                            pos_tol=config.pos_tol, rb_tol=config.rb_tol, ang_tol=config.ang_tol)
            self.pool.append(job)

    def sbatch_parameters(self, job):
        """
        Return sbatch command parameters for a job
        """
        now = datetime.now().strftime('%m-%d-%y_%H-%M-%S')
        sbatch_params = ['--partition', 'upex', '--job_name', job.name,
                         '--output', os.path.join(job.out_path,
                                                  self.output_file.format(job_name=job.name, now=now)),
                         '--error', os.path.join(job.out_path,
                                                 self.error_file.format(job_name=job.name, now=now))]
        return sbatch_params

    def batch_job(self, job, test):
        """
        Batch a job
        """
        command = [self.batch_cmd]
        command.extend(self.sbatch_parameters(job))
        command.extend([self.shell_script, self.geom_file, self.rb_file])
        command.extend(job.shell_parameters())
        print('Submitting job: {:s}'.format(job.name))
        print('Shell script: {:s}'.format(self.shell_script))
        print('Command: {:s}'.format(' '.join(command)))
        if test:
            return -1
        else:
            try:
                output = subprocess.run(args=command, check=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                err_text = self.error_text.format(cmd=' '.join(command),
                                                  code=error.returncode,
                                                  stderr=error.stderr)
                raise RuntimeError(err_text) from error
            job_num = int(output.stdout.rstrip().decode("unicode_escape").split()[-1])
            print("The job {} has been submitted".format(job.name))
            print("Job ID: {}".format(job_num))
            return job_num

    def batch(self, test=False):
        """
        Batch a pool of jobs
        """
        for job in self.pool:
            self.batch_job(job, test)

def main():
    parser = argparse.ArgumentParser(description='Batch to Maxwell jobs of indexing refinement')
    parser.add_argument('config_file', type=str, help='Path to a config ini file')
    parser.add_argument('geom_file', type=str, help='Path to an experimental geometry ini file')
    parser.add_argument('rb_file', type=str, help='Path to a reciprocal lattice basis vectors ini file')
    parser.add_argument('--test', action='store_true', help='Test batching the job to the Maxwell cluster')
    args = parser.parse_args()

    batcher = JobBatcher(args.config_file, args.geom_file, args.rb_file)
    batcher.batch(test=args.test)
