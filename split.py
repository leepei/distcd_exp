#!/usr/bin/env python

import sys, subprocess, uuid, os, math, shutil

if len(sys.argv) != 3 and len(sys.argv) != 5:
    print('usage: {0} [-f NFS_flag (default 0)] machinefile svm_file'.format(sys.argv[0]))
    sys.exit(1)
machinefile_path, src_path = sys.argv[-2:]

NFS_flag = 0

if len(sys.argv) == 5:
    if sys.argv[1] == '-f':
        NFS_flag = int(sys.argv[2])
    else:
        print('usage: {0} [-f NFS_flag (default 0)] machinefile svm_file'.format(sys.argv[0]))
        sys.exit(1)

machines = set()
for line in open(machinefile_path):
    machine = line.strip()
    if machine in machines:
        print('Ignoring duplicated machine {0}'.format(machine))
    else:
        machines.add(machine)

nr_machines = len(machines)

dst_path = ''
src_basename = os.path.basename(src_path)
if NFS_flag == 0:
    dst_path = '{0}.sub'.format(src_basename)

cmd = 'wc -l {0}'.format(src_path)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
nr_instances = int(p.stdout.read().strip().split()[0])
p.communicate()


if (NFS_flag == 0):
    while True:
        temp_dir = 'tmp_{0}'.format(uuid.uuid4())
        if not os.path.exists(temp_dir): break
    os.mkdir(temp_dir)
    split_path = os.path.join(temp_dir, src_basename)
else:
    split_path = src_basename

print('Spliting data...')
nr_digits = int(math.log10(nr_machines))+1
cmd = 'split -l {0} --numeric-suffixes -a {1} {2} {3}.'.format(
          int(math.ceil(float(nr_instances)/nr_machines)), nr_digits, src_path,
          split_path)
p = subprocess.Popen(cmd, shell=True)
p.communicate()

if (NFS_flag == 0):
    print('Sending data...')
    for i, machine in enumerate(machines):
        print i, machine
        temp_path = os.path.join(temp_dir, src_basename + '.' + str(i).zfill(nr_digits))
        if machine == '127.0.0.1' or machine == 'localhost':
            cmd = 'mv {0} {1}'.format(temp_path, dst_path)
        else:
            cmd = 'scp {0} {1}:{2}'.format(temp_path, machine, os.path.join(os.getcwd(), dst_path))
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p.communicate()

    shutil.rmtree(temp_dir)
print('Task done')
