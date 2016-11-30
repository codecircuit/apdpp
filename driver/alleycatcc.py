#!/usr/bin/env python3
"""
Alleycat compiler driver

Copyright (c) 2015, All Rights Reserved

Author: Alexander Matz
Affiliation: ZITI, Heidelberg University
License: GPL (BSD might be better, but less safe)

This compiler driver consists of two pipelines, one for BSP code
and one for host code (if a frontend comes up, a CUDA pipeline
might be added), with seperate binaries.

Pipeline stages:

    1               2                  3                 4                5
 compile         analyze           transform           link            assemble

BSP:
[cl -> ir]   [ir -> ir + db]   [ir + db -> ir]  [ir + libclc -> ir]   [ir -> ptx]

Host:
[c++ -> ir]  [ir -> ir + db]   [ir + db -> ir]  [ir + obj -> obj]     [obj -> bin]
"""

from distutils.spawn import find_executable
import os
import os.path
from os import getenv
import argparse
import sys
import subprocess as sp
import tempfile
import traceback

tools = {}
libs = {}
config = {}

def log(vbs, msg):
    if vbs <= config['verbose']:
        print(msg)

# we want to hardwire paths to match with libs
llvm_tools_binary_dir = getenv('LLVM_TOOLS_BINARY_DIR')
tools['clang'] = llvm_tools_binary_dir + '/clang'
tools['clang++'] = llvm_tools_binary_dir + '/clang++'
tools['opt'] = llvm_tools_binary_dir + '/opt'
tools['llc'] = llvm_tools_binary_dir + '/llc'
tools['llvm-link'] = llvm_tools_binary_dir + '/llvm-link'
tools['llvm-dis'] = llvm_tools_binary_dir + '/llvm-dis'
# system included linker required
tools['ld'] = find_executable('ld')

def get_argparser():
    parser = argparse.ArgumentParser(description='Alleycat compiler driver', add_help=False)
    parser.add_argument('-h', '-help', action='help', help='Show this help message')
    parser.add_argument('-###', action='store_true', dest='dry', help='Print commands, but do not run')
    parser.add_argument('-c', action='store_true', help='only execute compile/analyze/transform stages')
    parser.add_argument('-S', action='store_true', help='only execute compile stage (same as -c -fno-analyze -fno-transform')
    parser.add_argument('-emit-llvm', action='store_true', help='emit llvm ir instead of bytecode, not possible after linking')
    parser.add_argument('-host-out', action='store', metavar='file', help='host code output filename')
    parser.add_argument('-bsp-out', action='store', metavar='file', help='bsp code output filename')
    parser.add_argument('-host-in', action='store', metavar='file', nargs='+', help='input files for host pipeline')
    parser.add_argument('-bsp-in', action='store', metavar='file', nargs='+', help='input files for bsp pipeline')
    parser.add_argument('-db', action='store', metavar='file', help='analysis database filename')
    parser.add_argument('-v', '-verbose', action='count', dest='verbose', default=0, help='verbose output')
    parser.add_argument('-bI', action='append', dest='bsp_incd', metavar='path', help='add BSP include path')
    parser.add_argument('-bL', action='append', dest='bsp_libd', metavar='path', help='add BSP library path')
    parser.add_argument('-l', action='append', dest='host_lib', metavar='path', help='add host library name'+\
                                                                                '(e.g. name=cuda refers to file=libcuda.so)')
    parser.add_argument('-hI', action='append', dest='host_incd', metavar='path', help='add host include path')
    parser.add_argument('-hL', action='append', dest='host_libd', metavar='path', help='add host library path')
    parser.add_argument('-fno-analyze', action='store_false', dest='analyze',
                        help='disable Static Code Analysis')
    parser.add_argument('-fno-transform', action='store_false', dest='transform',
                        help='disable Transformations code')
    return parser

def get_tempfname(suffix):
    return tempfile._get_default_tempdir()+'/'+next(tempfile._get_candidate_names())+'.'+suffix

def is_of(fname, types):
    fu = fname.lower()
    for t in types:
        if fu.endswith(t):
            return True
    return False

class Unit:
    tmp_files = []
    def clean_tmp_files():
        for f in Unit.tmp_files:
            if os.path.isfile(f):
                os.remove(f)
    def __init__(self, filename):
        self.filename = filename
        self.lower = self.filename.lower()
        self.basename = os.path.basename(self.filename)
        self.name, self.ext = os.path.splitext(self.basename)
        self.fnames = []
        self.actions = []
    def exists(self):
        return os.path.isfile(self.fname)
    def add_action(self, action):
        self.actions += [action]
    def last_action(self):
        if len(self.actions) < 1:
            return None
        else:
            return self.actions[-1]
    def add_fname(self, fname):
        self.fnames+= [fname]
        return fname
    def add_tmp(self, ext):
        name = tempfile._get_default_tempdir()+'/'+next(tempfile._get_candidate_names())+'-'+self.name+ext
        if config['verbose']:
            print("adding temporary file", name)
        self.fnames += [name]
        Unit.tmp_files += [name]
        return name
    def last_in(self):
        if len(self.fnames) > 1:
            return self.fnames[-2]
        else:
            return self.filename
    def last_out(self):
        if len(self.fnames) > 0:
            return self.fnames[-1]
        else:
            raise Exception('No output file assigned for unit \'%s\'' % (self.name))
    def any_of(self, ftypes):
        for t in ftypes:
            if t == self.ext:
                return True
    def last_any_of(self, ftypes):
        for t in ftypes:
            _, ext = os.path.splitext(self.last_in())
            if t == ext:
                return True
        return False

class Action:
    def __init__(self, kind, args):
        self.kind = kind
        self.result = None
        self.cmd = []
        self.args = args
        # args: Unit (in+out)
        if kind == 'cl-compile':
            # checks
            if len(args) != 1:
                raise Exception('Wrong number of arguments')

            self.cmd = [tools["clang"]]
            self.cmd += ['-Dcl_clang_storage_class_specifiers']
            self.cmd += ['-x', 'cl']
            if 'libclc_inc' in config:
                self.cmd += ['-isystem', config['libclc_inc']]
            self.cmd += ['-include', 'clc/clc.h']
            self.cmd += ['-target', config['bsp_target']]
            self.cmd += ['-c']
            self.cmd += ['-emit-llvm']
            for p in config['bsp_incd']:
                self.cmd += ['-I', p]
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (in+out)
        elif kind == 'bsp-ll-compile':
            # checks
            if len(args) != 1:
                raise Exception('Wrong number of arguments')

            self.cmd = [tools["clang"]]
            self.cmd += ['-target', config['bsp_target']]
            self.cmd += ['-c']
            self.cmd += ['-emit-llvm']
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (in+out)
        elif kind == 'bsp-analyze':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['opt']]
            self.cmd += ['-load', 'lib/bsp_analysis.so']
            self.cmd += ['-bsp_analysis']
            self.cmd += ['-alleycat_db', config['db']]
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (in+out)
        elif kind == 'bsp-transform':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['opt']]
            self.cmd += ['-load', './lib/bsp_transform.so']
            self.cmd += ['-bsp_transform']
            self.cmd += ['-alleycat_db', config['db']]
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (out), input units
        elif kind == 'bsp-link':
            if len(args) < 2:
                raise Exception('Not enough arguments')
            self.cmd = [tools['llvm-link']]
            self.cmd += ['-o', args[0].last_out()]
            for unit in args[1:]:
                self.cmd += [unit.last_out()]
            libdevice = config['bsp_target']+'.bc'
            if 'libclc_lib' in config:
                self.cmd += [config['libclc_lib']+'/'+libdevice]
            else:
                self.cmd += [libdevice]
        # args: Unit (in+out)
        elif kind == 'bsp-asm':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['clang']]
            self.cmd += ['-target', config['bsp_target']]
            self.cmd += ['-S']
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (in+out)
        elif kind == 'disasm':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['llvm-dis']]
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        # args: Unit (in+out)
        elif kind == 'cp':
            self.cmd = ['cp']
            self.cmd += [args[0].last_in()]
            self.cmd += [args[0].last_out()]
        elif kind == 'cc-compile':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['clang++']]
            self.cmd += ['-c']
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
            self.cmd += ['-I', config['cuda_inc']] # as we use cuda driver api
            self.cmd += ['-I', config['mekong_inc']] # mekong include files relative to this driver script, alleycat.h depends on e.g. on bitop and json
            self.cmd += ['-include', 'alleycat.h']
            self.cmd += ['-std=c++0x'] # c++11
            self.cmd += ['-emit-llvm']
            for p in config['host_incd']:
                self.cmd += ['-I', p]
        elif kind == 'host-ll-compile':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['clang++']]
            self.cmd += ['-S']
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
            self.cmd += ['-emit-llvm']
        elif kind == 'host-transform':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['opt']]
            self.cmd += ['-load', './lib/host_transform.so']
            self.cmd += ['-host_transform']
            self.cmd += ['-alleycat_db', config['db']]
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        elif kind == 'host-link':
            if len(args) < 2:
                raise Exception('not enough arguments')
            self.cmd = [tools['clang++']]
            self.cmd += ['-o', args[0].last_in()]
            for unit in args[1:]:
                self.cmd += [unit.last_out()]
            self.cmd += ['-L', config['cuda_lib'], '-lcuda']
            self.cmd += ['-L', './lib', '-ljsoncpp', '-lbitop']
            for p in config['host_libd']:
                self.cmd += ['-L', p]
        elif kind == 'host-asm':
            if len(args) != 1:
                raise Exception('Wrong number of arguments')
            self.cmd = [tools['clang++']]
            self.cmd += ['-S']
            self.cmd += ['-o', args[0].last_out()]
            self.cmd += [args[0].last_in()]
        else:
            raise Exception('Unsupported action type: \'%s\'' % (kind))
    def execute(self):
        if self.result is None:
            if config['dry']:
                log(0, " ".join(self.cmd))
                self.result = True
                return self.result
            else:
                self.result = sp.call(self.cmd) == 0
                return self.result
        else:
            return self.result
    def __str__(self):
        return '[%s -> %s, %s]' % (self.ifiles, self.ofiles, self.kind)

def config_static():
    conf = {}
    conf['bsp_target'] = 'nvptx64--nvidiacl'
    conf['mekong_inc'] = './inc'
    return conf

def config_environment():
    env_config = {}
    if getenv('LIBCLC_INC'):
        env_config['libclc_inc'] = getenv('LIBCLC_INC')
    if getenv('LIBCLC_LIB'):
        env_config['libclc_lib'] = getenv('LIBCLC_LIB')
    if getenv('CUDA_INC'):
        env_config['cuda_inc'] = getenv('CUDA_INC')
    if getenv('CUDA_LIB'):
        env_config['cuda_lib'] = getenv('CUDA_LIB')
    return env_config

def main():
    global config
    config = config_static()
    config.update(config_environment())
    args = get_argparser().parse_args()
    config.update(vars(args))

    # normalize array arguments
    config['bsp_in'] = config['bsp_in'] or []
    config['bsp_out'] = config['bsp_out'] or []
    config['host_in'] = config['host_in'] or []
    config['host_out'] = config['host_out'] or []
    config['bsp_incd'] = config['bsp_incd'] or []
    config['bsp_libd'] = config['bsp_libd'] or []
    config['host_incd'] = config['host_incd'] or []
    config['host_libd'] = config['host_libd'] or []

    log(2, config)

    # stage selection

    stages = [False] * 6
    # compile
    stages[0] = True
    # analyze, 'S' = only execute compile stage (same as -c -fno-analyze -fno-transform)
    stages[1] = not config['S'] and config['analyze']
    # transform
    stages[2] = not config['S'] and config['transform']
    # link, 'c' = only execute compile/analyze/transform stages
    stages[3] = not config['S'] and not config['c']
    # assemble
    stages[4] = (not config['S']) and (not config['c']) and (not config['emit_llvm'])
    # disassemble
    stages[5] = (not stages[3]) and (not stages[4]) and config['emit_llvm']
    #FIXME correct functionality of last stages
    stages[4] = False
    stages[5] = False
    print("CAUTION: ASSEMBLE AND DISASSEMBLE STAGE ARE DEACTIVATED AND PTX BACKEND AIMS AT sm_20")

    last_stage = -1
    if stages[5]:
        last_stage = 5
    elif stages[4]:
        last_stage = 4
    elif stages[3]:
        last_stage = 3
    elif stages[2]:
        last_stage = 2
    elif stages[1]:
        last_stage = 1
    elif stages[0]:
        last_stage = 0

    log(2, 'Stages: %s' % stages)
    log(2, 'Last stage: %s' % last_stage)
    if config['verbose']:
        print("Last stage =", last_stage)

    # Check for valid configuration flags

    bsp_active = len(config['bsp_in']) > 0
    host_active = len(config['host_in']) > 0

    if len(config['bsp_in']) < 1 and len(config['host_in']) < 1:
        raise Exception('No input files')

    if config['bsp_out'] and len(config['bsp_in']) < 1:
        raise Exception('Can\'t produce bsp output without bsp input files')

    if len(config['bsp_in']) > 0 and not config['bsp_out']:
        raise Exception('No output file for bsp pipeline specified')

    if len(config['host_in']) > 0 and not config['host_out']:
        raise Exception('No output file for host pipeline specified')

    if (stages[1] or stages[2]) and not config['db']:
        raise Exception('Analysis/transformation passes require a database file')

    if config['host_out'] and len(config['host_in']) < 1:
        raise Exception('Can\'t produce host output without host input files')

    if not stages[3] and (len(config['bsp_in']) > 1 or len(config['host_in']) > 1):
        raise Exception('Multiple input files not valid without linking stage')

    if (stages[3] or stages[4]) and config['emit_llvm']:
        raise Exception('Can\'t emit llvm ir after linking')

    log(1, 'Files in BSP pipeline: %s' % (' '.join(config['bsp_in'] or [] )))
    log(1, 'Files in host pipeline: %s' % (' '.join(config['host_in'] or [] )))

    actions = [None]*6
    actions[0] = [] # compile
    actions[1] = [] # analyze
    actions[2] = [] # transform
    actions[3] = [] # link
    actions[4] = [] # assemble
    actions[5] = [] # assemble

    # Collect translation units

    # bsp pipeline units
    bsp_units = []
    for f in config['bsp_in']:
        bsp_units += [Unit(f)]

    # host pipeline units
    host_units = []
    for f in config['host_in']:
        host_units += [Unit(f)]

    # combined pipeline units, created as needed
    bsp_comb_unit = None
    host_comb_unit = None

    # Compile stage
    if stages[0]:
        if bsp_active:
            for unit in bsp_units:
                if last_stage == 0:
                    unit.add_fname(config['bsp_out'])
                else:
                    if (config['emit_llvm']):
                        unit.add_tmp('.ll')
                    else:
                        unit.add_tmp('.bc')
                if unit.last_any_of(['.cl']):
                    action = Action('cl-compile', [unit])
                    unit.add_action(action)
                    actions[0] += [action]
                elif unit.last_any_of(['.bc', '.ll']):
                    action = Action('bsp-ll-compile', [unit])
                    unit.add_action(action)
                    actions[0] += [action]
                else:
                    raise Exception('Unknown file format of \'%s\'' % unit.filename)
        if host_active:
            for unit in host_units:
                if last_stage == 0:
                    unit.add_fname(config['host_out'])
                else:
                    if (config['emit_llvm']):
                        unit.add_tmp('.ll')
                    else:
                        unit.add_tmp('.bc')
                if unit.last_any_of(['.cc', '.cpp']):
                    action = Action('cc-compile', [unit])
                    unit.add_action(action)
                    actions[0] += [action]
                elif unit.last_any_of(['.bc', '.ll']):
                    action = Action('host-ll-compile', [unit])
                    unit.add_action(action)
                    actions[0] += [action]
                else:
                    raise Exception('Unknown file format of \'%s\'' % unit.filename)

    # Analyze stage

    if stages[1]:
        if bsp_active:
            for unit in bsp_units:
                if last_stage == 1:
                    unit.add_fname(config['bsp_out'])
                else:
                    unit.add_tmp('.bc')
                if unit.last_any_of(['.bc', '.ll']):
                    action = Action('bsp-analyze', [unit])
                    unit.add_action(action)
                    actions[1] += [action]
                else:
                    raise Exception('Unknown file format of \'%s\'' % unit.filename)
    # TODO host part

    # Transform stage

    if stages[2]:
        if bsp_active:
            for unit in bsp_units:
                if last_stage == 2:
                    unit.add_fname(config['bsp_out'])
                else:
                    unit.add_tmp('.bc')
                if unit.last_any_of(['.bc', '.ll']):
                    action = Action('bsp-transform', [unit])
                    unit.add_action(action)
                    actions[2] += [action]
                else:
                    raise Exception('Unknown file format of \'%s\'' % unit.filename)
        if host_active:
            for unit in host_units:
                if last_stage == 2:
                    unit.add_fname(config['host_out'])
                else:
                    unit.add_tmp('.bc')
                if unit.last_any_of(['.bc', '.ll']):
                    action = Action('host-transform', [unit])
                    unit.add_action(action)
                    actions[2] += [action]
                else:
                    Exception('Unknown file format of \'%s\'' % unit.filename)

    # Link stage

    if stages[3]:
        if bsp_active:
            if bsp_comb_unit is None:
                bsp_comb_unit = Unit(config['bsp_out'])
            if last_stage == 3:
                bsp_comb_unit.add_fname(config['bsp_out'])
            else:
                if (config['emit_llvm']):
                    bsp_comb_unit.add_tmp('.ll')
                else:
                    bsp_comb_unit.add_tmp('.bc')
            action = Action('bsp-link', [bsp_comb_unit] + bsp_units)
            bsp_comb_unit.add_action(action)
            actions[3] += [action]
        if host_active:
            if host_comb_unit is None:
                print("host comb unit is none")
                host_comb_unit = Unit(config['host_out'])
            if last_stage == 3:
                host_comb_unit.add_fname(config['host_out'])
            else:
                if (config['emit_llvm']):
                    host_comb_unit.add_tmp('.ll')
                else:
                    host_comb_unit.add_tmp('.bc')
            print("host comb unit last in =", host_comb_unit.last_in())
            print("host comb unit last out =", host_comb_unit.last_out())
            action = Action('host-link', [host_comb_unit] + host_units)
            host_comb_unit.add_action(action)
            actions[3] += [action]

    # Assemble stage

    if stages[4]:
        if bsp_active:
            if bsp_comb_unit is None:
                bsp_comb_unit = Unit(config['bsp_out'])
            if last_stage == 4:
                bsp_comb_unit.add_fname(config['bsp_out'])
                action = Action('bsp-asm', [bsp_comb_unit])
                bsp_comb_unit.add_action(action)
                actions[4] += [action]
            else:
                raise Exception('Something went wrong, linking should be '+\
                                'last step when active')
        if host_active:
            print("assemble stage:")
            if host_comb_unit is None:
                host_comb_unit = Unit(config['host_out'])
            if last_stage == 4:
                host_comb_unit.add_fname(config['host_out'])
                action = Action('host-asm', [host_comb_unit])
                host_comb_unit.add_action(action)
                actions[4] += [action]
            else:
                raise Exception('Something went wrong, linking should be '+\
                                'last step when active')

    # Aux stage

    if stages[5]:
        if bsp_active:
            if last_stage == 5:
                unit = bsp_units[0]
                unit.add_fname(config['bsp_out'])
                action = Action('disasm', [unit])
                unit.add_action(action)
                actions[5] += [action]
            else:
                raise Exception('Something went wrong, disassembly should be '+\
                                'last step when active')
        if host_active:
            if last_stage == 5:
                unit = host_units[0]
                unit.add_fname(config['host_out'])
                action = Action('disasm', [unit])
                unit.add_action(action)
                actions[5] += [action]
            else:
                raise Exception('Something went wrong, disassembly should be '+\
                                'last step when active')

    # Execution
    for stage in actions:
        for action in stage:
            result = action.execute()
            if not result:
                raise Exception('Subcommand failed')

    return 0

main()

"""if __name__ == "__main__":
    result = 0
    try:
        result = main()
    except Exception as e:
        print("Error: %s" % (e.args[0]))
        result = 1
        #tb = traceback.format_exc()
        print(tb)
    finally:
        Unit.clean_tmp_files()
        sys.exit(result)"""
