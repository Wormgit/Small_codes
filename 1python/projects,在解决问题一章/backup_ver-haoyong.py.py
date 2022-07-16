import os
import time

# 1. The files and directories to be backed up are
# specified in a list.
source = ['/home/io18230/Desktop/Jing/qita','/home/io18230/Desktop/Jing/好用的','/home/io18230/Desktop/pictures']
target_dir = '/home/io18230/Desktop/s'


# 3. The files are backed up into a zip file.
# 4. The name of the zip archive is the current date and time
target = target_dir + os.sep + \
         time.strftime('%Y%m%d%H%M%S') + '.zip'

# Create target directory if it is not present
if not os.path.exists(target_dir):
    os.mkdir(target_dir)  # make directory

# 5. We use the zip command to put the files in a zip archive
zip_command = 'zip -r {0} {1}'.format(target,
                                      ' '.join(source))

# Run the backup
print(zip_command)
print('Running:')
if os.system(zip_command) == 0:
    print('Successful backup to', target)
else:
    print('Backup FAILED')
