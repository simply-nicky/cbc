"""
index.py - run and batch to Maxwell cluster indexing refinement jobs
"""
import argparse
import subprocess
import os
from timeit import default_timer as timer
from datetime import datetime
import h5py
import numpy as np
from ..utils import INIParser, OUT_PATH, PROJECT_PATH, FILENAME, chunkify, make_path
from ..model import RecBasis
from ..feat_detect import ScanSetup
from ..indexer import ScanStreaks

class IndexJob(INIParser):
    """
    Frame index  refinement job parameters class

    mode                                - indexing refinement mode
    out_path                            - output filename
    scan_num                            - scan number
    pop_size                            - refinement population size
    n_isl                               - number of refinement islands
    gen_num                             - number of refinement algorithm generations
    (f_tol, smp_tol, rb_tol, ang_tol)   - focus, sample coordinates,
                                          reciporcal basis vector lengths,
                                          and rotation angles tolerances
    """
    section = 'config'
    fn_str = "{scan_num:03d}_{mode:s}_index"
    index_keys = ('n_isl', 'pop_size', 'gen_num', 'f_tol', 'smp_tol', 'rb_tol', 'ang_tol')

    def __init__(self, **kwargs):
        self.data_dict = kwargs
        out_dir = os.path.join(PROJECT_PATH, OUT_PATH['scan'].format(scan_num=kwargs['scan_num']), 'index')
        os.makedirs(out_dir, exist_ok=True)
        if kwargs['filename']:
            filename = kwargs['filename']
        else:
            filename = self.fn_str.format(scan_num=kwargs['scan_num'],
                                                mode=kwargs['mode']) + '.h5'
        self.out_path = make_path(os.path.join(out_dir, filename))
        self.setup = ScanSetup.import_ini(kwargs['geom_file'])
        self.scan_path = os.path.join(OUT_PATH['scan'].format(scan_num=kwargs['scan_num']),
                                      FILENAME['scan'].format(tag='streaks', ext='h5',
                                                              scan_num=kwargs['scan_num']))
        self.index_dict = {key:kwargs[key] for key in self.index_keys}
        self.index_dict['rec_basis'] = RecBasis.import_ini(kwargs['rb_file'])

    @classmethod
    def import_ini(cls, ini_file):
        """
        Import from an ini file

        config_file - path to a file
        """
        ini_parser = cls.read_ini(ini_file)
        data_dict = dict(ini_parser[cls.section])
        data_dict['scan_num'] = ini_parser.getint(cls.section, 'scan_num')
        data_dict['n_isl'] = ini_parser.getint(cls.section, 'n_isl')
        data_dict['pop_size'] = ini_parser.getint(cls.section, 'pop_size')
        data_dict['gen_num'] = ini_parser.getint(cls.section, 'gen_num')
        data_dict['f_tol'] = ini_parser.getfloatarr(cls.section, 'f_tol')
        data_dict['smp_tol'] = ini_parser.getfloatarr(cls.section, 'smp_tol')
        data_dict['rb_tol'] = ini_parser.getfloat(cls.section, 'rb_tol')
        data_dict['ang_tol'] = ini_parser.getfloat(cls.section, 'ang_tol')
        if data_dict['mode'] == 'scan':
            data_dict['frames'] = ini_parser.getintarr(cls.section, 'frames')
        return data_dict

    def scan(self):
        """
        Return ScanStreaks class object
        """
        print("Looking for data file: {}".format(self.scan_path))
        if not os.path.exists(self.scan_path):
            raise ValueError("Data doesn't exist at the following path: {}".format(self.scan_path))
        print("Opening the data file...")
        with h5py.File(self.scan_path, 'r') as data_file:
            scan = ScanStreaks(raw_lines=data_file['streaks/lines'][:],
                               exp_set=self.setup,
                               frame_idxs=data_file['streaks/frame_idxs'][:])
        print("{:d} streaks detected in total".format(scan.size))
        return scan

    def run_index(self):
        """
        Run indexing refinement
        """
        scan = self.scan()
        print("Setting up the indexing solution refinement...")
        if self.mode == 'rot':
            archi = scan.rot_index(**self.index_dict)
        else:
            archi = scan.full_index(**self.index_dict)
        print("Starting indexing solution refinement...")
        start = timer()
        archi.evolve()
        archi.wait()
        print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
        index_sol = np.array(archi.get_champions_x()).reshape((self.n_isl, scan.frames.size, -1), order='F')
        index_pts = np.array(archi.get_champions_f()).reshape((self.n_isl, scan.frames.size), order='F')
        self.write(index_sol, index_pts)

    def write(self, index_sol, index_pts):
        """
        Write refinement solutions to an hdf5 file

        index_sol - an array of refinement solutions
        index_pts - an array of refinement points
        """
        with h5py.File(self.out_path, 'w') as out_file:
            for key in self.index_keys:
                out_file.create_dataset(os.path.join(self.section, key), data=self.data_dict[key])
            out_file.create_dataset(os.path.join(self.section, 'rb_mat'),
                                    data=self.index_dict['rec_basis'].rb_mat)
            out_file.create_dataset('data/index_sol', data=index_sol)
            out_file.create_dataset('data/index_pts', data=index_pts)
            print("The refined solutions have been saved, file path: {}".format(self.out_path))

class ScanIndexJob(IndexJob):
    """
    Scan index refinement job parameters class

    out_path                            - output filename
    scan_num                            - scan number
    pop_size                            - refinement population size
    n_isl                               - number of refinement islands
    gen_num                             - number of refinement algorithm generations
    (f_tol, smp_tol, rb_tol, ang_tol)   - focus, sample coordinates,
                                          reciporcal basis vector lengths,
                                          and rotation angles tolerances
    frames                              - list of frame indices to refine
    """

    def run_index(self):
        """
        Run indexing refinement
        """
        scan = self.scan()[self.frames]
        print("Setting up the indexing solution refinement...")
        archi = scan.scan_index(**self.index_dict)
        print("Starting indexing solution refinement...")
        start = timer()
        archi.evolve()
        archi.wait()
        print("The refinement has been completed, elapsed time: {:f}s".format(timer() - start))
        ev_pops = [island.get_population() for island in archi]
        index_sol = np.concatenate([pop.get_x() for pop in ev_pops])
        index_pts = np.concatenate([pop.get_f() for pop in ev_pops])
        self.write(index_sol, index_pts)

class Batcher():
    """
    sbatch jobs to cluster

    config_file - path to a config ini file
    geom_file - path to an experimental geometry ini file
    """
    batch_cmd, frmt = "sbatch", '%m-%d-%y_%H-%M-%S'
    index_script = os.path.join(PROJECT_PATH, "cbc_dp/index/index.sh")
    combine_script = os.path.join(PROJECT_PATH, "cbc_dp/index/combine.sh")
    job_name = "{scan_num:03d}_{mode:s}_index_part{idx:02d}"
    fn_str = "{scan_num:03d}_{mode:s}_index"
    out_file = "{job_name:s}_{now:s}.out"
    err_file = "{job_name:s}_{now:s}.err"
    error_text = "Command '{cmd:s}' has returned an error (code {code:s}): {stderr:s}"
    job_keys = ('geom_file', 'rb_file', 'mode', 'scan_num', 'pop_size',
                'gen_num', 'f_tol', 'smp_tol', 'rb_tol', 'ang_tol', 'frames')
    job_size = 16

    def __init__(self, **kwargs):
        self.params = kwargs
        self.out_dir = os.path.join(PROJECT_PATH, OUT_PATH['scan'].format(scan_num=kwargs['scan_num']), 'index')
        os.makedirs(self.out_dir, exist_ok=True)
        self.sb_dir = os.path.join(PROJECT_PATH, OUT_PATH['scan'].format(scan_num=kwargs['scan_num']), 'sbatch_out')
        os.makedirs(self.sb_dir, exist_ok=True)
        if kwargs['filename']:
            filename = kwargs['filename']
        else:
            filename = self.fn_str.format(scan_num=kwargs['scan_num'],
                                          mode=kwargs['mode']) + '.h5'
        self.out_path = make_path(os.path.join(self.out_dir, filename))
        self._init_pool()

    def _init_pool(self):
        self.pool_job = {}
        for idx, n_isl in enumerate(chunkify(self.n_isl, self.job_size)):
            job_name = self.job_name.format(scan_num=self.scan_num, mode=self.mode, idx=idx)
            params = {key:self.params[key] for key in self.job_keys if key in self.params}
            params['filename'] = job_name + '.h5'
            params['n_isl'] = n_isl
            self.pool_job[job_name] = params

    @classmethod
    def now(cls):
        """
        Return current date and time string at the particular format
        """
        return datetime.now().strftime(cls.frmt)

    @staticmethod
    def shell_parameters(params):
        """
        Return shell paramseters for a job
        """
        shell_params = [params['geom_file'], params['rb_file'], params['mode']]
        for key, item in params.items():
            if item != None:
                if key in ['f_tol', 'smp_tol', 'frames']:
                    shell_params.extend(['--' + key, str(item).strip('[]')])
                else:
                    shell_params.extend(['--' + key, str(item)])
        return shell_params

    def __getattr__(self, attr):
        if attr in self.params:
            return self.params[attr]

    def sbatch_parameters(self, job_name):
        """
        Return sbatch command parameters for a job
        """
        sbatch_params = ['--partition', 'upex', '--job-name', job_name, '--chdir', PROJECT_PATH,
                         '--time', '4-00:00:00',
                         '--output', os.path.join(self.sb_dir,
                                                  self.out_file.format(job_name=job_name,
                                                                       now=self.now())),
                         '--error', os.path.join(self.sb_dir,
                                                 self.err_file.format(job_name=job_name,
                                                                      now=self.now()))]
        return sbatch_params

    def index_command(self, job_name, params):
        """
        Return a command to batch an indexing job
        """
        command = [self.batch_cmd]
        command.extend(self.sbatch_parameters(job_name))
        command.extend(['--mail-type', 'END,FAIL', self.index_script])
        command.extend(self.shell_parameters(params))
        return command

    def combine_command(self, job_nums):
        """
        Return a command to batch a combine job
        """
        command = [self.batch_cmd]
        command.extend(self.sbatch_parameters('combine'))
        command.extend(['--dependency',
                        'afterok:{:s}'.format(':'.join(job_nums)),
                        self.combine_script])
        command.extend([os.path.join(self.out_dir, params['filename'])
                        for params in self.pool_job.values()])
        command.append(self.out_path)
        return command

    def batch_job(self, job_name, command):
        """
        Batch a job
        """
        print('Submitting job: {:s}'.format(job_name))
        print('Command: {:s}'.format(' '.join(command)))
        if self.test:
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

    def batch(self):
        """
        Batch a pool of jobs
        """
        job_nums = []
        for job_name, params in self.pool_job.items():
            command = self.index_command(job_name, params)
            job_nums.append(self.batch_job(job_name, command))
        command = self.combine_command(job_nums)
        self.batch_job('combine', command)

def main():
    """
    Main fuction to run and batch indexing refinement jobs
    """
    parser = argparse.ArgumentParser(description='Run CBC indexing refinement')
    parser.add_argument('geom_file', type=str, help='Path to a geometry ini file')
    parser.add_argument('rb_file', type=str,
                        help='Path to a reciprocal lattice basis vectors ini file')
    parser.add_argument('--mode', type=str, choices=['rot', 'full', 'scan'], default='rot',
                        help='Choose between rotation and full indexing refinement')
    parser.add_argument('--scan_num', type=int, default=135, help='Scan number')
    parser.add_argument('--pop_size', type=int, default=50,
                        help='Population size of the refinement islands')
    parser.add_argument('--n_isl', type=int, default=16, help='Number of islands for one frame')
    parser.add_argument('--gen_num', type=int, default=3000,
                        help='Maximum generations number of the refinement algorithm')
    parser.add_argument('--f_tol', type=float, nargs=3, default=[2e-2, 2e-2, 1e-4])
    parser.add_argument('--smp_tol', type=float, nargs=3, default=[5e-2, 5e-2, 5e-2],
                        help='Relative sample position tolerance')
    parser.add_argument('--rb_tol', type=float, default=1e-3,
                        help='Lattice basis vectors length tolerance')
    parser.add_argument('--ang_tol', type=float, default=1e-1, help='Rotation anlges tolerance')
    parser.add_argument('--frames', type=int, nargs='+', help='Frames to index (scan mode only)')
    parser.add_argument('--ini_file', type=str, help='Open an ini file to fetch configuration parameters')
    parser.add_argument('--filename', type=str, help='Output file name')
    parser.add_argument('--batch', action='store_true', help='Batch the job to Maxwell cluster')
    parser.add_argument('--test', action='store_true', help='Test batching the job to Maxwell cluster')

    params = vars(parser.parse_args())
    if params['ini_file']:
        params.update(IndexJob.import_ini(params['ini_file']))

    if params['batch']:
        Batcher(**params).batch()
    else:
        if params['mode'] == 'scan':
            ScanIndexJob(**params).run_index()
        else:
            IndexJob(**params).run_index()
