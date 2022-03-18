#!/usr/bin/env python
# coding: utf-8

#%%==================================================
'''
import dependencies
'''
import subprocess
from random import randint
import random
import warnings
import socket
# gives us access to global configurations, e.g. what is used as VLAN interface
from common import *

#%%==================================================
'''
macro & global variables
'''
CONF = {}  # configuration of testbed
NODES = {}  # a dictionary of all node instances of different type
LOOP_DIR = '/mnt/loop'
VERBOSE = False
LB_METHOD = 'maglev'
LB_METHOD_LAST_BUILD = 'maglev'
dirname = os.path.dirname(__file__)

#%%==================================================
'''
lower-level utils
'''

'''
pretty print
'''


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    def print_info(self, line):
        print(self.BLUE + self.BOLD +
              ">> {}".format(line)
              + self.END)


class FileNotExistWarning(UserWarning):
    pass


'''
execute command and print out result
'''


def subprocess_cmd(command):
    global VERBOSE
    '''
    pretty print
    '''
    def pretty_print(_str, _color="green"):
        if (len(_str) != 0):
            print(colored(_str, _color))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    if VERBOSE:
        pretty_print(proc_stdout.decode("utf-8"))
    return proc_stdout.decode("utf-8")


#%%==================================================
'''
kernel-level operation functions
'''

'''
mount new image: img_orig -> img_new
@param:
    img_orig: original image
    img_new: target 
'''


def mount_new_image(img_orig, img_new):
    global LOOP_DIR
    if VERBOSE:
        print(">> Mounting image: {}...".format(img_new), end='\r')
    cmd = "rm -f {0};\
    qemu-img create -f qcow2 -b {1} {0};\
    sudo modprobe nbd max_part=8;\
    sudo qemu-nbd --connect=/dev/nbd0 {0};\
    sudo mount -o loop /dev/nbd0p1 {2};\n".format(img_new, img_orig, LOOP_DIR)
    subprocess_cmd(cmd)
    if VERBOSE:
        print(">> Mounting image {} done!".format(img_new))


'''
mount image: img
@param:
    img: image to be mounted
@e.g.:
    mount_image(CONFIG['img']['base'])
    # then run `sudo chroot /mnt/loop/ /bin/bash` to check something like:
    # `apt list --installed | grep vpp`
'''


def mount_image(img):
    global LOOP_DIR
    if VERBOSE:
        print(">> Mounting image: {}...".format(img), end='\r')
    cmd = "sudo modprobe nbd max_part=8;\
    sudo qemu-nbd --connect=/dev/nbd0 {};\
    sudo mount -o loop /dev/nbd0p1 {};\
    \n".format(img, LOOP_DIR)
    subprocess_cmd(cmd)
    if VERBOSE:
        print(">> Mounting image done!")


'''
copy files from src to dst
@param:
    src/dst: path on the machine
'''


def copy_files(src, dst, isfolder=False):
    if VERBOSE:
        print("- Copy from {} to {}...".format(src, dst), end='\r')
    r_placeholder = ' '
    if isfolder:
        r_placeholder = ' -r '
    cmd = "sudo cp{}{} {}".format(r_placeholder, src, dst)
    subprocess_cmd(cmd)
    if VERBOSE:
        print("- Copy from {} to {} done!".format(src, dst))


'''
cleanup mounted image (umount and disconnect)
'''


def umount_image():
    global LOOP_DIR
    if os.path.exists('{}/home/cisco'.format(LOOP_DIR)):
        if VERBOSE:
            print(">> Umounting image...")
        cmd = "sudo chown 1000:1000 {0}/home/cisco/*;\
        sudo umount {0}/;\
        sudo qemu-nbd --disconnect /dev/nbd0;\n".format(LOOP_DIR)
        subprocess_cmd(cmd)
        if VERBOSE:
            print(">> Umounting image done!")


'''
write content to filename
@param:
    content: string
    filename: in form of os.path
    attach_mode: by default turn off
                 change to '-a' s.t. content will be attached to the end of the file
    debug_mode: by default turn off
                change to whatever s.t. content will be printed out
'''


def write2file_tee(content, filename, attach_mode=False, debug_mode=None):
    if attach_mode:
        attach = '-a '
    else:
        attach = ''
    if debug_mode:
        cmd = 'echo \"{}" | sudo tee {}{}'.format(content, attach, filename)
    else:
        cmd = 'echo \"{}" | sudo tee {}{} > /dev/null'.format(
            content, attach, filename)
    subprocess_cmd(cmd)


'''
randomly generate l2 address
'''


def generateL2Addr():
    return "de:ad:ca:fe:{:02x}:{:02x}".format(randint(0, 255), randint(0, 255))


'''
take down all tap interfaces
@param:
    tap_list: list of tap interface names
'''


def tap_down(tap_list):
    cmd = ""
    for tap in tap_list:
        cmd += "sudo tunctl -d {};".format(tap)
    subprocess_cmd(cmd)


'''
bring up all tap interfaces
@param:
    tap_list: list of tap interface names
'''


def tap_up(tap_list):
    cmd = ""
    for tap in tap_list:
        cmd += "sudo tunctl -t {0}; sudo ifconfig {0} mtu 1500 up;".format(tap)
    subprocess_cmd(cmd)


'''
setup bridge if needed
'''


def br_up():
    global CONF
    '''
    setup 802.1q mode and necessary interfaces for bridge
    '''
    def if_up():
        cmd = "sudo modprobe 8021q;\
        sudo ifconfig {0} up;\
        sudo vconfig add {0} {1};\
        sudo ifconfig {0}.{1} up;\
        sudo vconfig add {0} {2};\
        sudo ifconfig {0}.{2} up;".format(COMMON_CONF["net"]["vlan_if"], COMMON_CONF["net"]["vlan_id"], COMMON_CONF["net"]["vlan_mgmt_id"])
        subprocess_cmd(cmd)
    bridge = CONF['global']['net']['bridge']
    mgmt_bridge = CONF['global']['net']['mgmt_bridge']
    if_up()
    # if the bridge does not exists
    if ~isfile(join("/sys/devices/virtual/net", bridge)):
        cmd = "sudo brctl addbr {0};\
        sudo brctl setageing {0} 9999999;\
        sudo ifconfig {0} up;\
        sudo brctl addif {0} {2}.{3};\
        sudo brctl addbr {1};\
        sudo brctl setageing {1} 9999999;\
        sudo ifconfig {1} up;\
        sudo brctl addif {1} {2}.{4};".format(bridge, mgmt_bridge, COMMON_CONF["net"]["vlan_if"], COMMON_CONF["net"]["vlan_id"], COMMON_CONF["net"]["vlan_mgmt_id"])
        subprocess_cmd(cmd)


'''
tear down bridge 
'''


def br_down():
    global CONF
    '''
    setup 802.1q mode and necessary interfaces for bridge
    '''
    def if_down():
        cmd = "sudo ifconfig {0} down;\
        sudo ifconfig {0}.{1} down;\
        sudo ip link del {0}.{1};\
        sudo ifconfig {0}.{2} down;\
        sudo ip link del {0}.{2};".format(COMMON_CONF["net"]["vlan_if"], COMMON_CONF["net"]["vlan_id"], COMMON_CONF["net"]["vlan_mgmt_id"])
        subprocess_cmd(cmd)
    bridge = CONF['global']['net']['bridge']
    mgmt_bridge = CONF['global']['net']['mgmt_bridge']
    cmd = "ls /sys/devices/virtual/net/{}/brif | grep -q -v {}.{}".format(
        bridge, COMMON_CONF["net"]["vlan_if"], COMMON_CONF["net"]["vlan_id"])
    if not subprocess_cmd(cmd):
        cmd = "sudo brctl delif {0} {2}.{3};\
        sudo ifconfig {0} down;\
        sudo brctl delbr {0};\
        sudo brctl delif {1} {2}.{4};\
        sudo ifconfig {1} down;\
        sudo brctl delbr {1};".format(bridge, mgmt_bridge, COMMON_CONF["net"]["vlan_if"], COMMON_CONF["net"]["vlan_id"], COMMON_CONF["net"]["vlan_mgmt_id"])
        subprocess_cmd(cmd)
        if_down()


'''
pin qemu thread who queries cpu to single cpu
'''


def pin_qemu(port, *argv):
    cmd = 'TIDS=$({\n\
cat << EOF\n\
{ "execute": "qmp_capabilities" }\n\
{ "execute": "query-cpus" }\n\
EOF\n\
}' + ' \
| nc localhost {} 2>&1 | tee tmp.txt > /dev/null & sleep 0.1;\
 kill -9 $( pgrep nc | tail -n 1 );\
 cat tmp.txt | tail -n 1 | jq -r ".return[].thread_id");\
rm -f tmp.txt;\
echo $TIDS;'.format(port)
    a = subprocess_cmd(cmd)
    for i, tid in enumerate(a.split(' ')):
        cmd = "sudo taskset -cp {} {};".format(argv[i], tid)
        subprocess_cmd(cmd)


#%%==================================================
'''
create base image
'''

'''
install vpp packages
'''


def install_vpp():
    global LOOP_DIR, CONF
    if VERBOSE:
        print(">> Install vpp...")
    cmd = "#!/bin/bash\n\
sudo chroot {}/ /bin/bash << EOF\n\
dpkg -i /home/cisco/libvppinfra_*.deb\n\
dpkg -i /home/cisco/vpp_*.deb\n\
dpkg -i /home/cisco/vpp-dbg_*.deb\n\
apt -y remove vpp-dev\n\
dpkg -i /home/cisco/libvppinfra-dev_*.deb\n\
dpkg -i /home/cisco/vpp-dev_*.deb\n\
dpkg -i /home/cisco/vpp-api-python_*.deb\n\
dpkg -i /home/cisco/vpp-plugin-dpdk_*.deb\n\
dpkg -i /home/cisco/vpp-plugin-core_*.deb\n\
dpkg -i /home/cisco/python3-vpp-api_*.deb\n\
service vpp disable\n\
update-rc.d vpp remove\n\
echo --------------------------------\n\
echo +++ Check installed packages +++\n\
apt list --installed | grep vpp\n\
exit\n\
EOF".format(LOOP_DIR)

    filename = join(CONF['global']['path']['tmp'], "install_vpp.sh")
    with open(filename, "w") as text_file:
        text_file.write(cmd)

    subprocess_cmd("sudo chmod +x {}".format(filename))
    subprocess_cmd("sudo bash {}".format(filename))
    subprocess_cmd("sudo rm -f {}".format(filename))
    if VERBOSE:
        print(">> Install vpp done!")


'''
pip install in base image
'''


def pip_install_base():
    global LOOP_DIR, CONF
    if VERBOSE:
        print(">> pip install...")
    cmd = "#!/bin/bash\n\
sudo chroot {}/ /bin/bash << EOF\n\
pip install psutil gym torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html\n\
exit\n\
EOF".format(LOOP_DIR)

    filename = join(CONF['global']['path']['tmp'], "install_pip.sh")
    with open(filename, "w") as text_file:
        text_file.write(cmd)

    subprocess_cmd("sudo chmod +x {}".format(filename))
    subprocess_cmd("sudo {}".format(filename))
    subprocess_cmd("sudo rm -f {}".format(filename))
    if VERBOSE:
        print(">> Install pip done!")


'''
move vpp plugin files to base image
'''


def install_vpp_plugin(plugin_list, vpp_dir):
    global LOOP_DIR
    if VERBOSE:
        print(">> Install vpp plugins...", end='\r')
    # uncomment the following line if you want to clean up stale plugins from .deb
#     subprocess_cmd("sudo rm {}/usr/lib/vpp_plugins/srlb_*.so".format(LOOP_DIR))
    for plugin in plugin_list:
        copy_files(join(vpp_dir, "{}_plugin.so".format(plugin)),
                   join(LOOP_DIR, "usr/lib/vpp_plugins"))
    if VERBOSE:
        print(">> Install vpp plugins done!")


'''
- mount base image
- copy custom files from backup dir
- copy vpp debs
- install vpp
- [install vpp plugins]
- remove deb files
- cleanup (unmount)
'''


def create_base_image():
    global CONF, LOOP_DIR, LB_METHOD, LB_METHODS
    # make sure there is no mounted image
    umount_image()
    # mount base image
    mount_new_image(CONF['global']['path']['orig_img'],
                    CONF['global']['path']['base_img'].replace('lb-', LB_METHODS[LB_METHOD]['img_id']+'-'))
    # copy vpp debs
    copy_files(join(CONF['global']['path']['vpp'], '*.deb'),
               join(LOOP_DIR, "home/cisco/"))
    # copy vpp plugins
    copy_files(join(CONF['global']['path']['vpp'], '*.so'),
               join(LOOP_DIR, "usr/lib/vpp_plugins/"))

    # update /etc/gai.conf so that it will be easier to install stuffs
    filename = join(LOOP_DIR, 'etc/gai.conf')
    write2file_tee('precedence ::ffff:0:0/96  100',
                   filename, attach_mode=False)

    # update /etc/gai.conf so that it will be easier to install stuffs
    filename = join(LOOP_DIR, 'home/cisco/lb_method')
    write2file_tee(LB_METHOD,
                   filename, attach_mode=False)

    # install vpp plugins
    install_vpp()
    # remove deb files
    cmd = "sudo rm {}/home/cisco/*.deb".format(LOOP_DIR)
    subprocess_cmd(cmd)
    # install tensorflow, psutil
    # pip_install_base()
    # cleanup
    umount_image()


#%%==================================================
'''
node classes
'''

'''
father class
'''


class Node(object):
    global CONF, LOOP_DIR, LB_METHOD, LB_METHODS

    def __init__(self, **kwargs):
        self.__dict__.update({k: v for k, v in kwargs.items()})
        self.backup_dir = join(CONF['global']['path']['src'], self.node_type)

    # branch a new image from base image
    def mount_new_image(self):
        umount_image()  # make sure there is no mounted image
        mount_new_image(CONF['global']['path']['base_img'].replace('lb-', LB_METHODS[LB_METHOD]['img_id']+'-'), self.img)

    # mount image of this node
    def mount_image(self):
        if isfile(self.img):  # if image file exists
            mount_image(self.img)
        else:
            warnings.warn('Image {} does not exist.'.format(
                self.img), FileNotExistWarning)

    # umount image
    def umount_image(self):
        umount_image()

    # update file
    def update_file(self, content, filename, attach=False):
        filename = LOOP_DIR + filename
        write2file_tee(content, filename, attach_mode=attach)

    # copy file
    def copy_file(self, src, dst, isfolder=False):
        dst = LOOP_DIR + dst
        copy_files(src, dst, isfolder=isfolder)

    # create a folder
    def create_folder(self, folder_dir):
        cmd = "sudo mkdir {}/{}".format(LOOP_DIR, folder_dir)
        subprocess_cmd(cmd)

    # create a folder via ssh
    def create_folder_ssh(self, folder_dir):
        cmd = "ssh -t -t -i ~/.ssh/lb_rsa cisco@localhost -p {} \"sudo mkdir {}\"".format(
            self.ssh_port, folder_dir)
        subprocess_cmd(cmd)

    # update file via ssh
    def update_file_ssh(self, content, filename, attach=False):
        # just to make sure that the content doesn't mess up with the cmd
        assert not '"' in content or not "'" in content
        if attach:
            attach = '-a '
        else:
            attach = ''
        cmd = 'ssh -t -t -i ~/.ssh/lb_rsa cisco@localhost -p {} "echo \'{}\' | sudo tee {}{} > /dev/null"'.format(
            self.ssh_port,
            content,
            attach,
            '~/{}'.format(filename)
        )
        subprocess_cmd(cmd)

    # copy file via scp
    def copy_file_scp(self, src, dst, isfolder=False):
        r_placeholder = ' '
        if isfolder:
            r_placeholder = ' -r '
        cmd = 'scp -i ~/.ssh/lb_rsa -oStrictHostKeyChecking=no -P {}{}{} cisco@{}:{} ;'.format(
            self.ssh_port, r_placeholder, src, self.physical_server_ip, '~/{}'.format(dst))
        subprocess_cmd(cmd)

    # execute cmd locally via ssh
    def execute_cmd_ssh(self, cmd):
        cmd = 'ssh -t -t -i ~/.ssh/lb_rsa cisco@localhost -p {} "{}"'.format(
            self.ssh_port,
            cmd
        )
        subprocess_cmd(cmd)

    # update universal files

    def update_files_univ(self):
        # update /etc/network/interfaces
        _content = "# This file describes the network interfaces available on your system\n\
# and how to activate them. For more information, see interfaces(5).\n\
\n\
source /etc/network/interfaces.d/*\n\
\n\
# The loopback network interface\n\
auto lo\n\
iface lo inet loopback\n\
\n\
# The primary network interface\n\
auto eth0\n\
iface eth0 inet dhcp\n\
\n\
# Out-of-band management interface\n\
auto eth1\n\
iface eth1 inet static\n\
address {}\n\
netmask 24\n\
\n".format(self.mgmt_ip)
        if not self.isvpp:
            _content += "# Data-plane interface\n\
auto eth2\n\
iface eth2 inet static\n\
address {0}\n\
netmask {1}\n\
".format(self.ip4_list[0], self.sn4_list[0])
            ip_list = self.ip4_list[1:] + self.ip6_list
            sn_list = self.sn4_list[1:] + self.sn6_list
            for ip, sn in zip(ip_list, sn_list):
                _content += "up ifconfig eth2 add {}/{}\n".format(ip, sn)
        self.update_file(_content, '/etc/network/interfaces')

        # update /etc/hostname
        _content = self.hostname
        self.update_file(_content, '/etc/hostname')

        # update /etc/hosts
        _content = "127.0.0.1       localhost\n\
# 127.0.1.1       {}\n\
\n\
# The following lines are desirable for IPv6 capable hosts\n\
::1     localhost ip6-localhost ip6-loopback\n\
ff02::1 ip6-allnodes\n\
ff02::2 ip6-allrouters".format(self.hostname)
        self.update_file(_content, '/etc/hosts')

        # create /home/cisco/init.sh
        _content = "#!/bin/bash\n\
#sudo ntpdate ntp.esl.cisco.com\n"

        # for k, v in CONF['arp'].items():
        #     _content += 'sudo arp -s {} {}\n'.format(k, v)

        self.update_file(_content, '/home/cisco/init.sh')
        # make scripts executable
        cmd = "sudo chmod +x {}/home/cisco/*.sh".format(LOOP_DIR)
        subprocess_cmd(cmd)

    # update vpp files
    def update_files_vpp(self):
        if self.isvpp:
            # create /etc/vpp/startup.conf
            _content = "unix {\n\
nodaemon\n\
log /tmp/vpp.log\n\
full-coredump\n\
cli-listen /run/vpp/cli.sock\n\
startup-config /home/cisco/vpp.startup\n\
}\n\
\n\
api-trace {\n\
on\n\
}\n\
\n\
dpdk {\n\
socket-mem 1024\n\
}"

            self.update_file(_content, '/etc/vpp/startup.conf')

            # create /home/cisco/vpp.startup
            ip_list = self.ip4_list + self.ip6_list
            sn_list = self.sn4_list + self.sn6_list
            _content = ''
            for ip, sn in zip(ip_list, sn_list):
                _content += 'set int ip addr GigabitEthernet0/5/0 {0}/{1}\n'.format(
                    ip, sn)
            _content += 'set interface promisc off GigabitEthernet0/5/0\n\
set interface state GigabitEthernet0/5/0 up\n\
ip6 nd GigabitEthernet0/5/0 ra-suppress\n\n'
            self.update_file(_content, '/home/cisco/vpp.startup')
        else:
            pass

    # bootstrap universal files
    def bootstrap_univ(self):
        self.update_files_univ()
        self.update_files_vpp()

    def get_qemu_cmd(self, graphic='-nographic'):
        # configure the eth0 as a management interface (NAT device)
        res = " -device virtio-net-pci,netdev=nat,bus=pci.0,addr=3.0 -netdev user,id=nat,hostfwd=tcp::{:d}-:22".format(
            self.ssh_port)
        # configure the eth1 as an out-of-band interface (bind it to the management bridge)
        res += " -device virtio-net-pci,netdev=mgmt,bus=pci.0,addr=4.0,mac={} -netdev tap,id=mgmt,vhost=on,ifname={},script=/bin/true".format(
            self.l2_list[0], self.tap_list[0])
        # configure the remaining interfaces (which will be used by VPP) as supplied in the argument
        for i, addr in enumerate(self.l2_list[1:]):
            res += " -device virtio-net-pci,netdev=net{0},bus=pci.0,addr={1}.0,mac={2} -netdev tap,id=net{0},vhost=on,ifname={3},script=/bin/true".format(
                i, i+5, addr, self.tap_list[i+1])
        # final touch
        res = "sudo qemu-system-x86_64 -enable-kvm -cpu host -smp {3} -qmp tcp:localhost:1{0},server,nowait -m 4096 -drive file={1},cache=writeback,if=virtio {2}".format(
            self.ssh_port, self.img, "-k en" if graphic != "-nographic" else "-nographic", len(self.vcpu_list)) + res
        return res

    def run(self):
        '''
        spin up vm with qemu and pin cpu to specific two
        '''
        def qemu_run(cmd):
            subprocess.Popen(cmd, shell=True)
            time.sleep(3)
            pin_qemu(10000+self.ssh_port, *self.vcpu_list)
            time.sleep(1)
            pin_qemu(10000+self.ssh_port, *self.vcpu_list)
        tap_up(self.tap_list)
        cmd = self.get_qemu_cmd()
        qemu_run(cmd)
        print('{} ready: ssh -p {} cisco@localhost'.format(self.hostname, self.ssh_port))

    def poweroff(self):
        cmd = 'ssh -t cisco@localhost -p {} "sudo poweroff" 2> /dev/null'.format(
            self.ssh_port)
        subprocess_cmd(cmd)

    def shutdown(self):
        self.poweroff()
        time.sleep(5)
        tap_down(self.tap_list)


'''
lb node class
'''


class lbNode(Node):

    def __init__(self, conf_dict):
        super(self.__class__, self).__init__(**conf_dict)

    def __copy_script(self):
        for file in ['gt_socket_check.py', 'shm_layout.json', 'shm_proxy.py', 'env.py', 'log_usage.py'] + self.files2copy:
            super(self.__class__, self).copy_file(
                join(self.backup_dir, file), join('/home/cisco', file))
        super(self.__class__, self).update_file(
            self.init_file_content, '/home/cisco/init.sh', attach=True)

    def __bootstrap_local(self):
        # update /home/cisco/vpp.startup
        ## config gre tunnel
        cnt = 0
        _content = ''
        for i in range(len(self.as_list)):
            _content += "create gre tunnel src {0} dst {1}\n\
set int state gre{2:d} up\n\
set int ip address gre{2:d} {3}/24\n\
create gre tunnel src {4} dst {5}\n\
set int state gre{6:d} up\n\
set int ip address gre{6:d} {7}/64\n\n\
".format(self.ip4_list[0], self.as_ip4_list[i], 2*i, self.gre4_list[i],
                self.ip6_list[0], self.as_ip6_list[i], 2*i+1, self.gre6_list[i])
        ## config lb
#         _content += "lb conf ip4-src-address {0}\n\
# lb vip {1}/24 encap gre4 new_len 64\n\
# lb as {1}/24 {2}\n\
# lb conf ip6-src-address {3}\n\
# lb vip {4}/64 encap gre6 new_len 64\n\
# lb as {4}/64 {5}\n\n\
# ".format(self.ip4_list[0], self.vip4, ' '.join(self.as_ip4_list),
#          self.ip6_list[0], self.vip6, ' '.join(self.as_ip6_list))
        _content += "lb conf ip6-src-address {0}\n\
lb vip {1}/64 encap gre6 new_len 64\n\
lb as {1}/64 {2}\n\n\
".format(self.ip6_list[0], self.vip6, ' '.join(self.as_ip6_list))
        ## config route for v4
        for clt_ip, er_ip in zip(self.clt_ip4_list, self.er_ip4_list):
            _content += 'ip route add {}/32 via {} GigabitEthernet0/5/0\n'.format(
                clt_ip, er_ip)
        super(self.__class__, self).update_file(
            _content, '/home/cisco/vpp.startup', attach=True)
        self.__copy_script()

        # if 'rl' in LB_METHOD_LAST_BUILD or 'heuristic' in LB_METHOD_LAST_BUILD:
        ## create a folder to store some other results
        super(self.__class__, self).create_folder('/home/cisco/log')

        # put lb id info and total number of lb nodes in a file
        _content = "{}/{}/{}".format(self.id,
                                     len(NODES['lb']), CONF['global']['topo']['n_node']['as'])
        super(self.__class__, self).update_file(_content, '/home/cisco/topo')

    def get_qemu_cmd(self, graphic='-nographic'):
        # configure the eth0 as a management interface (NAT device)
        res = " -device virtio-net-pci,netdev=nat,bus=pci.0,addr=3.0 -netdev user,id=nat,hostfwd=tcp::{:d}-:22".format(
            self.ssh_port)
        # configure the eth1 as an out-of-band interface (bind it to the management bridge)
        res += " -device virtio-net-pci,netdev=mgmt,bus=pci.0,addr=4.0,mac={} -netdev tap,id=mgmt,vhost=on,ifname={},script=/bin/true".format(
            self.l2_list[0], self.tap_list[0])
        # configure the remaining interfaces (which will be used by VPP) as supplied in the argument
        for i, addr in enumerate(self.l2_list[1:]):
            res += " -device virtio-net-pci,netdev=net{0},bus=pci.0,addr={1}.0,mac={2} -netdev tap,id=net{0},vhost=on,ifname={3},script=/bin/true".format(
                i, i+5, addr, self.tap_list[i+1])
        # final touch
        res = "sudo qemu-system-x86_64 -enable-kvm -cpu host -smp {3} -qmp tcp:localhost:1{0},server,nowait -m 8192 -drive file={1},cache=writeback,if=virtio {2}".format(
            self.ssh_port, self.img, "-k en" if graphic != "-nographic" else "-nographic", len(self.vcpu_list)) + res
        return res

    def bootstrap(self):
        super(self.__class__, self).bootstrap_univ()
        self.__bootstrap_local()

    def create_img(self):
        print("Create LB node image...")
        super(self.__class__, self).mount_new_image()
        self.bootstrap()
        super(self.__class__, self).umount_image()

    def mount_img(self):
        super(self.__class__, self).mount_image()

    def umount_img(self):
        super(self.__class__, self).umount_image()

    def execute_cmd_ssh(self, cmd):
        super(self.__class__, self).execute_cmd_ssh(cmd)

    def run(self):
        super(self.__class__, self).run()


    def gather_usage(self):
        cmd = 'ssh -t -p {} cisco@localhost "sudo python3 log_usage.py &"'.format(
            self.ssh_port)
        subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    def gt_socket_check(self, n_server):
        cmd = 'ssh -t -p {} cisco@localhost "sudo python3 gt_socket_check.py {}"'.format(
            self.ssh_port, n_server)
        log = subprocess_cmd(cmd)[1:-1]
        if len(log) > 0:
            return [int(i) for i in log.split(',')]
        else:
            return []

    def run_init_bg(self):
        self.gather_usage()
        cmd = 'ssh -n -f cisco@localhost -p {} "sh -c \'cd ~/; nohup ./init.sh &\'" &'.format(
            self.ssh_port)
        subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        

    def fetch_result(self, dir, episode, filename='log', isfolder=True):
        '''
        @brief:
            scp results from LB node to host, store at ${dir}/shm.csv
        @patams:
            dir: directory to store file
        '''
        r_placeholder = ' '
        if isfolder:
            r_placeholder = ' -r '
        cmd = 'ssh -n -f cisco@localhost -p {} "grep @dt /var/log/syslog > ~/log/clib.log"'.format(self.ssh_port)
        subprocess_cmd(cmd)
        cmd = 'scp -i ~/.ssh/lb_rsa -oStrictHostKeyChecking=no -P {0}{1}cisco@{5}:~/{3} {2}/{4}_{3}_ep{6};'.format(
            self.ssh_port, r_placeholder, dir, filename, self.id, self.physical_server_ip, episode)
        subprocess_cmd(cmd)
        target_dir = cmd.split(' ')[-1].strip(';')
        return target_dir

    def shutdown(self):
        super(self.__class__, self).shutdown()


'''
application server node class
'''


class asNode(Node):

    def __init__(self, conf_dict):
        super(self.__class__, self).__init__(**conf_dict)

    def __bootstrap_local(self):
        # update /etc/network/interfaces
        ## config gre tunnel
        cnt = 0
        _content = ''
        for i in range(len(self.lb_list)):
            _content += "# config gre int for lb {0}\n\
auto gre{1}\n\
iface gre{1} inet static\n\
address {2}\n\
netmask 24\n\
pre-up ip tunnel add gre{1} mode gre local {3} remote {4} ttl 255\n\
post-down ip tunnel del gre{1}\n\
\n\
# config gre6 int for lb {0}\n\
up ip -6 tunnel add gre{5} mode ip6gre local {6} remote {7} ttl 255 encaplimit none\n\
up ip link set gre{5} up\n\
up ip -6 addr add {8}/64 dev gre{5}\n\
post-down ip -6 tunnel del gre{5}\n\n\
".format(i, 2*i+1, self.vip4, self.ip4_list[0], self.lb_ip4_list[i],
                2*i+2, self.ip6_list[0], self.lb_ip6_list[i], self.vip6
         )
        _content += ''  # config client route\n'
        for ip4 in self.clt_ip4_list:
            _content += 'up ip route add {}/32 dev gre1\n'.format(ip4)
        for ip6 in self.clt_ip6_list:
            _content += 'up ip -6 route add {}/128 dev eth2\n'.format(ip6)
        super(self.__class__, self).update_file(
            _content, '/etc/network/interfaces', attach=True)

        # get ground truth gatherer
        super(self.__class__, self).copy_file(
            join(self.backup_dir, 'shmlog'), '/home/cisco/shmlog')
        super(self.__class__, self).copy_file(
            join(self.backup_dir, 'shm_server.py'), '/home/cisco/shm_server.py')
        super(self.__class__, self).update_file(
            '[ -z \`pgrep shmlog\` ] && sudo /home/cisco/shmlog {0} & sleep 1.5\n sudo python3 /home/cisco/shm_server.py\n'.format(self.id), '/home/cisco/init.sh', attach=True)

        # put asid into a file
        super(self.__class__, self).update_file(
            '{}'.format(self.id), '/home/cisco/asid', attach=False)

    def bootstrap(self):
        super(self.__class__, self).bootstrap_univ()
        self.__bootstrap_local()

    def create_img(self):
        super(self.__class__, self).mount_new_image()
        self.bootstrap()
        super(self.__class__, self).umount_image()

    def mount_img(self):
        super(self.__class__, self).mount_image()

    def umount_img(self):
        super(self.__class__, self).umount_image()

    def run(self):
        super(self.__class__, self).run()
        time.sleep(10)
        self.init()

    def init(self):
        cmd = 'ssh -t -p {} cisco@localhost "sudo service vpp stop; sudo bash ./init.sh" 2> /dev/null'.format(
            self.ssh_port)
        subprocess.Popen(cmd, shell=True)

    def shutdown(self):
        super(self.__class__, self).shutdown()


'''
edge router node class
'''


class erNode(Node):

    def __init__(self, conf_dict):
        super(self.__class__, self).__init__(**conf_dict)

    def __bootstrap_local(self):
        # update /home/cisco/vpp.startup
        ## config gre tunnel
        cnt = 0
        _content = ''
        # add route
        for ip4 in self.lb_ip4_list:
            _content += 'ip route add {}/32 via {} {}\n'.format(
                self.vip4, ip4, 'GigabitEthernet0/5/0')  # v4
        for ip6 in self.lb_ip6_list:
            _content += 'ip route add {}/128 via {} {}\n'.format(
                self.vip6, ip6, 'GigabitEthernet0/5/0')  # v6
        super(self.__class__, self).update_file(
            _content, '/home/cisco/vpp.startup', attach=True)

    def bootstrap(self):
        super(self.__class__, self).bootstrap_univ()
        self.__bootstrap_local()

    def create_img(self):
        super(self.__class__, self).mount_new_image()
        self.bootstrap()
        super(self.__class__, self).umount_image()

    def mount_img(self):
        super(self.__class__, self).mount_image()

    def umount_img(self):
        super(self.__class__, self).umount_image()

    def run(self):
        super(self.__class__, self).run()

    def shutdown(self):
        super(self.__class__, self).shutdown()


'''
client node class
'''


class cltNode(Node):

    def __init__(self, conf_dict):
        super(self.__class__, self).__init__(**conf_dict)

    def __bootstrap_local(self):
        # update /etc/network/interfaces
        ## config gre tunnel
        cnt = 0
        _content = '# config ip route\n'
        for ip4 in self.er_ip4_list:
            _content += 'up ip route add {0}/32 via {1} dev eth2\n'.format(
                self.vip4, ip4)
        for ip6 in self.er_ip6_list:
            _content += 'up ip -6 route add {0}/128 via {1} dev eth2\n'.format(
                self.vip6, ip6)

        super(self.__class__, self).update_file(
            _content, '/etc/network/interfaces', attach=True)
        super(self.__class__, self).copy_file(
            join(self.backup_dir, 'replay_fork_io.py'), '/home/cisco/replay.py')
        super(self.__class__, self).copy_file(
            join(self.backup_dir, 'run_clt.sh'), '/home/cisco/run_clt.sh')

    def bootstrap(self):
        super(self.__class__, self).bootstrap_univ()
        self.__bootstrap_local()

    def create_img(self):
        super(self.__class__, self).mount_new_image()
        self.bootstrap()
        super(self.__class__, self).umount_image()

    def mount_img(self):
        super(self.__class__, self).mount_image()

    def umount_img(self):
        super(self.__class__, self).umount_image()

    def run(self):
        super(self.__class__, self).run()
        cmd = 'ssh -t -p {} cisco@localhost "sudo service vpp stop;"'.format(
            self.ssh_port)
        subprocess.Popen(cmd, shell=True)

    def startTraffic(self, bg=False):
        cmd = 'ssh -t -t cisco@localhost -p {} "sudo python3 replay.py trace.csv > trace.log"'.format(
            self.ssh_port)
        if bg:
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess_cmd(cmd)

    def fetch_result(self, dir, episode):
        '''
        @brief:
            scp results from Client node to host, store at ${dir}/trace.log
        @patams:
            dir: directory to store file
        '''
        cmd = 'scp -i ~/.ssh/lb_rsa -oStrictHostKeyChecking=no -P {} cisco@localhost:~/trace.log {}/trace.log;'.format(
            self.ssh_port, dir)
        subprocess_cmd(cmd)
        cmd = 'scp -oStrictHostKeyChecking=no {0}/trace.log yzy@{1}:{0}/trace_ep{2}.log;'.format(
            dir, COMMON_CONF['net']['base_ip'], episode)
        subprocess_cmd(cmd)

    def shutdown(self):
        super(self.__class__, self).shutdown()


#%%==================================================
'''
higher-level function
'''

'''
load config file for all nodes (generated by ${root}/src/utils/gen_conf.py)
Note: this should be the first function to call
@params:
    filename
'''


def get_config(filename):
    global COMMON_CONF, CONF
    conf_file = join(COMMON_CONF['dir']['root'], 'config', 'cluster', filename)
    if not os.path.isfile(conf_file):  # if config file doesn't exist
        warnings.warn(
            'Config {} does not exist, create w/ default config.'.format(conf_file), FileNotExistWarning)
    CONF = json_read_file(conf_file)


'''
get node instances given configuration
@params:
    node_config - configuration for different kinds of nodes (default = None and load global CONF as config)
                  e.g.:
    {'dev': [{'id': 0,
                ...
            }],
     'clt': [{'id': 0,
                ...
            }],
     'er': [{'id': 0,
                ...
            }],
     'lb': [{'id': 0,
                ...
            }],
     'as': [{'id': 0,
                ...
            },
            {'id': 1,
                ...
            }]}
                 
'''


def get_nodes(lb_method, node_config=None):
    global CONF, NODES, LB_METHODS, dirname
    # refresh LB_METHODS just in case of some updates
    filename = os.path.join(dirname, '../../config/lb-methods.json')
    LB_METHODS = json_read_file(filename)

    if not node_config:
        node_config = CONF['nodes']
    nodes = {}
    for k in node_config.keys():
        nodes[k] = []
        for node in node_config[k]:
            if k == 'lb':  # for LB nodes, add files to copy
                node['files2copy'] = LB_METHODS[lb_method]['files']
                node['init_file_content'] = '\n'.join(
                    LB_METHODS[lb_method]['init_lines'])
            nodes[k].append(eval(k+'Node')(node))
    NODES = nodes
    return nodes


'''
rebuild images for given nodes
@params:
    nodes - a dictionary of node instances generated from get_nodes() (default = None and load global NODES)
    from_orig - whether or not rebuild base image (default = False)
'''


def rebuild(nodes=None, from_orig=False):
    global NODES
    if not nodes:
        nodes = NODES
    if from_orig:
        create_base_image()
    for k in nodes.keys():
        for node in nodes[k]:
            node.create_img()


'''
setup bridge on the host side
@params:
    nodes - a dictionary of node instances generated from get_nodes() (default = None and load global NODES)
'''


def host_br_up(nodes=None):
    global CONF, NODES
    if not nodes:
        nodes = NODES
    # bring up bridges
    br_up()
    # get all tap interfaces (respectively for vpp and mgmt)
    tapvpp = []
    tapmgmt = []
    for k in nodes.keys():
        for node in nodes[k]:
            for tap in node.tap_list:
                if 'vpp' in tap:
                    tapvpp.append(tap)
                elif 'mgmt' in tap:
                    tapmgmt.append(tap)
    cmd = ''
    for tap in tapvpp:
        cmd += 'sudo brctl addif {0} {1};'.format(
            CONF['global']['net']['bridge'], tap)
    for tap in tapmgmt:
        cmd += 'sudo brctl addif {0} {1};'.format(
            CONF['global']['net']['mgmt_bridge'], tap)
    subprocess_cmd(cmd)


'''
run all the nodes
@params:
    nodes - a dictionary of node instances generated from get_nodes() (default = None and load global NODES)
'''


def runall(nodes=None):
    umount_image()
    if not nodes:
        nodes = NODES
    for k in nodes.keys():
        for node in nodes[k]:
            node.run()
    host_br_up()


'''
shutdown all the nodes
'''


def shutall(nodes=None):
    if not nodes:
        nodes = NODES
    tap_ifs = []
    for k in nodes.keys():
        for node in nodes[k]:
            node.poweroff()
            tap_ifs += node.tap_list
    time.sleep(5)
    tap_down(tap_ifs)
    br_down()


'''
check ground truth gathering socket connection
'''


def gt_socket_check(nodes=None):
    if not nodes:
        nodes = NODES
    for lb in nodes['lb']:
        err = lb.gt_socket_check(len(lb.as_list))
        while len(err) > 0:
            print("LB Node {}: found error socket with server {}".format(lb.id, err))
            time.sleep(1)
            for i in err:
                # best effort
                if i < len(nodes['as']):
                    nodes['as'][i].init()
            err = lb.gt_socket_check(len(lb.as_list))
        print("LB Node {}: pass".format(lb.id))
    if VERBOSE:
        print("GT socket check pass.")


def get_task_name_dir(experiment, lb_method, trace, sample, colocate=None, colocate_freq=0.0001, alias=None):
    '''
    @brief:
        initialize task name and corresponding directory to store results
    @params:
        n_lb: number of load balancer nodes
        n_as: number of application server nodes
        experiment: name of the high-level set of experiments
        lb_method: name of the load balancing method
        trace: type of the networking trace to be replayed
        sample: sample of the chosen type of trace
    @return:
        task_name: name of the task
        task_dir: directory to store experiment result for this task
        
    '''
    assert lb_method in LB_METHODS.keys()
    root_dir = COMMON_CONF['dir']['root']
    result_dir = join(root_dir, 'data', 'results')
    trace_dir = join(result_dir, experiment, trace)
    method_dir = join(trace_dir, lb_method)
    # set task name
    task_name = sample.rstrip(".csv")
    print('alias={}'.format(alias))
    if alias:
        task_name += '-{}'.format(alias)
    if colocate:
        # assert colocate < n_as
        task_name += '-{}co{:d}'.format(colocate, int(1/colocate_freq))
    task_dir = join(method_dir, task_name)
    task_name = '-'.join([trace, lb_method, task_name])
    return task_name, task_dir


def init_task_info(experiment, lb_method, trace, sample, n_lb=2, n_as=12, filename=None, colocate=None, colocate_freq=0.0001, alias=None):
    '''
    @brief:
        initialize configuration info w/ arguments
        setup task name
        setup directories to store experiment results w/ format: `${root}/data/results/${trace}/${lb_method}/${task}+${alias}/`
        setup node configuration for the cluster
    @params:
        n_lb: number of load balancer nodes
        n_as: number of application server nodes
        lb_method: name of the load balancing method
        trace: type of the networking trace to be replayed
        sample: sample of the chosen type of trace
    @return:
        task_name: name of the task
        task_dir: directory to store experiment result for this task
        nodes: configuration for all the nodes
    '''
    global CONF, COMMON_CONF, LB_METHOD
    assert lb_method in LB_METHODS.keys()
    LB_METHOD = lb_method
    root_dir = COMMON_CONF['dir']['root']
    result_dir = join(root_dir, 'data', 'results')
    experiment_dir = join(result_dir, experiment)
    trace_dir = join(experiment_dir, trace)
    method_dir = join(trace_dir, lb_method)
    print('init_task_info: alias={}'.format(alias))
    task_name, task_dir = get_task_name_dir(
        experiment, lb_method, trace, sample, colocate, colocate_freq, alias)

    # create folders if not yet existed
    dir2mk = [result_dir, experiment_dir, trace_dir, method_dir, task_dir]
    create_folder(dir2mk)

    if not filename:
        filename = '1clt-1er-{}lb-{}as.json'.format(n_lb, n_as)

    # get cluster config
    # create one if not yet existed
    if not os.path.exists(join(root_dir, 'config', 'cluster', filename)):
        cmd = "python3 {} --n-lb {} --n-as {}".format(os.path.join(
            root_dir, 'src', 'utils', 'gen_conf.py'), n_lb, n_as)
        subprocess_cmd(cmd)

    # load config
    get_config(filename)
    nodes = get_nodes(lb_method)

    # update shm_layout.json
    shm_layout = json_read_file(os.path.join(
        COMMON_CONF['dir']['root'], 'src', 'lb', 'shm_layout_base.json'))
    shm_layout.update({
        'meta': {
            'n_as': CONF['global']['topo']['n_node']['as'],
            'weights': CONF['global']['topo']['n_vcpu']['as']
        }
    })
    json_write2file(shm_layout, os.path.join(
        COMMON_CONF['dir']['root'], 'src', 'lb', 'shm_layout.json'))

    return task_name, task_dir, nodes

#--- Pipeline ---#


def prepare_img(lb_method, from_orig=None, debug_node=False):
    '''
    @brief:
        prepare all the image files
    @params:
        lb_method: name of the load balancing method
        from_orig
    '''
    global LB_METHODS, COMMON_CONF, LB_METHOD_LAST_BUILD, CONF

    assert lb_method in LB_METHODS.keys()
    root_dir = COMMON_CONF['dir']['root']
    vpp_dir = COMMON_CONF['dir']['vpp']

    if from_orig is None:  # if `from_orig` is not specified here
        if os.path.exists(CONF['global']['path']['base_img'].replace('lb-', LB_METHODS[lb_method]['img_id']+'-')):
            from_orig = False
        else:
            from_orig = True
            print("base img for {} does not exist, create one...".format(lb_method))

    if from_orig:
        # build vpp
        cmd = 'python3 {} -m {}'.format(join(root_dir,
                                             'src', 'vpp', 'gen_layout.py'), lb_method)
        if debug_node:
            cmd += ' -dn'
        subprocess_cmd(cmd)
        # copy generated lb files to vpp folder
        cmd = 'sudo cp -r {} {}/src/plugins/'.format(
            join(root_dir, 'src', 'vpp', 'lb'), vpp_dir)
        subprocess_cmd(cmd)
        cmd = 'sudo bash {}'.format(
            join(root_dir, 'src', 'vpp', 'lb-build.sh'))
        subprocess_cmd(cmd)
        cmd = 'cp {}/build-root/*.deb {}/data/vpp_deb/'.format(
            vpp_dir, root_dir)
        subprocess_cmd(cmd)
        cmd = 'cp {}/build-root/build-vpp-native/vpp/lib/vpp_plugins/lb_plugin.so {}/data/vpp_deb/'.format(
            vpp_dir, root_dir)
        subprocess_cmd(cmd)
        LB_METHOD_LAST_BUILD = lb_method

    umount_image()
    rebuild(from_orig=from_orig)


def run_colocate_workload(n_colocate, frequency=0.0001):
    '''
    @brief:
        start running some colocated workloads on $n_colocate$ servers with ${frequency}
    @params:
        n_colocate: number of servers on which co-located workloads are applied
        frequency: higher frequency requires more resource utilization (default frequency -> ~40% CPU)   
    '''
    global NODES
    assert n_colocate < len(NODES['as'])

    for i in range(n_colocate):
        as_id = len(NODES['as'])-i-1
        NODES['as'][as_id].copy_file_scp(
            join(NODES['as'][as_id].backup_dir, "colocate.py"), "colocate.py")
        cmd = 'ssh -t -p {} cisco@localhost "python3 colocate.py {}"'.format(
            NODES['as'][as_id].ssh_port, frequency)
        subprocess.Popen(cmd, shell=True)


def prepare_trace_sample(trace, sample, clip_n=None):
    '''
    @brief:
        scp trace sample files to client node
    @params:
        trace: type of the networking trace to be replayed
        sample: sample of the chosen type of trace
        clip_n: set a number if we want to clip off some lines and use only first ${clip_n} lines
    '''
    global CONF, NODES
    sample_file_dir = join(CONF['global']['path']['trace'], trace, sample)
    print(">> prepare trace sample:", sample_file_dir)
    if not os.path.exists(sample_file_dir):
        print("ERROR: sample does not exist:", sample_file_dir)
        return
    NODES['clt'][0].copy_file_scp(sample_file_dir, "sample.csv")
    if clip_n:
        cmd = 'ssh -t -t -i ~/.ssh/lb_rsa cisco@{} -p 8800 "head -n {} sample.csv > trace.csv; rm -f sample.csv"'.format(
            NODES['clt'][0].physical_server_ip, clip_n)
        subprocess_cmd(cmd)
    else:
        cmd = 'ssh -t -t -i ~/.ssh/lb_rsa cisco@{} -p 8800 "mv sample.csv trace.csv"'.format(
            NODES['clt'][0].physical_server_ip)
        subprocess_cmd(cmd)
