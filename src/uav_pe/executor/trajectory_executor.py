import json
import numpy as np
import time
import airsim
import os
import sys
import cv2
import random
import threading
import copy
import socket
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import threading

try:
    from tornado.iostream import StreamClosedError
    TORNADO_AVAILABLE = True
except ImportError:
    class StreamClosedError(Exception):
        pass
    TORNADO_AVAILABLE = False

tqdm.set_lock(threading.RLock())

def safe_log(msg, scene_id=None):
    if scene_id:
        msg = f"[{scene_id}] {msg}"
    tqdm.write(msg, file=sys.stderr)

try:
    import msgpackrpc
    MSGPACKRPC_AVAILABLE = True
except ImportError:
    MSGPACKRPC_AVAILABLE = False
    print("Warning: msgpackrpc is not installed; auto scene startup will be unavailable")

try:
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    root_logger.propagate = False
    
    from .logger import logger
    if hasattr(logger, 'logger'):
        for handler in logger.logger.handlers[:]:
            logger.logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.INFO)
        logger.logger.propagate = False
    USE_LOGGER = True
except:
    class SimpleLogger:
        @staticmethod
        def info(msg): print(f"[INFO] {msg}")
        @staticmethod
        def warning(msg): print(f"[WARNING] {msg}")
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}")
    logger = SimpleLogger()
    USE_LOGGER = False


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        if not MSGPACKRPC_AVAILABLE:
            raise RuntimeError("msgpackrpc is not installed; auto scene startup is unavailable")

        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        all_confirmed = True
        max_retries = 10
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    if USE_LOGGER:
                        logger.info(f"Start confirming connection: clients[{index_1}][{index_2}]")
                    while not confirmed and count < max_retries:
                        try:
                            self.airsim_clients[index_1][index_2].ping()
                            confirmed = True
                            if USE_LOGGER:
                                logger.info(f"Connection confirmed: clients[{index_1}][{index_2}]")
                        except Exception as e:
                            if count < 3 or count % 5 == 0:
                                if USE_LOGGER:
                                    logger.warning(f"Connection confirm failed (attempt {count + 1}): {str(e)}")
                            count += 1
                            if count >= max_retries:
                                if USE_LOGGER:
                                    logger.error(f"Connection confirm failed after {max_retries} retries: clients[{index_1}][{index_2}]")
                                all_confirmed = False
        
        return all_confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients
        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass
        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int = 180) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=600)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients

        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f"Start opening scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}")
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                error_detail = result[1] if len(result) > 1 and result[1] is not None else 'No error detail'
                logger.error(f"Failed to open scenes, machine: {socket_client.address._host}:{socket_client.address._port}")
                logger.error(f"Error detail: {error_detail}")
                raise Exception(f"Failed to open scenes: {error_detail}")
            assert len(result[1]) == 2, "Failed to open scenes"
            wait_time = 3 * len(self.machines_info[index]['open_scenes']) + 35
            if USE_LOGGER:
                logger.info("Waiting for scenes to start...")
            else:
                print(f'waiting for airsim connection...')
            ip = result[1][0]
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), "Failed to open scenes"
            assert len(ports) == len(self.machines_info[index]['open_scenes']), "Failed to open scenes"
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    if not USE_LOGGER:
                        print(f"AirSim client port: {port}")

            if USE_LOGGER:
                logger.info(f"Scenes opened, machine {index}: {socket_client.address._host}:{socket_client.address._port}")
            
            max_wait_time = 180
            check_interval = 5
            waited = 0
            ready = False

            port_check_interval = 2
            port_check_count = 0
            max_port_checks = 150

            if USE_LOGGER:
                logger.info(f"Waiting for port {ports[0]} to start listening...")
            port_listening = False
            scene_exited = False
            log_file_pattern = None
            try:
                from pathlib import Path
                log_dir = Path(__file__).parent.parent / 'core' / 'logs'
                if log_dir.exists() and len(self.machines_info[index]['open_scenes']) > 0:
                    scene_id = self.machines_info[index]['open_scenes'][0]
                    if scene_id is not None:
                        log_file_path = log_dir / f'scene_{scene_id}_{ports[0]}.log'
                        if log_file_path.exists():
                            log_file_pattern = log_file_path
                        else:
                            log_files = list(log_dir.glob(f'scene_*_{ports[0]}.log'))
                            if log_files:
                                log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                log_file_pattern = log_files[0]
            except:
                pass
            
            while port_check_count < max_port_checks:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((ip, ports[0]))
                    sock.close()
                    if result == 0:
                        if USE_LOGGER:
                            logger.info(f"Port {ports[0]} is listening")
                        port_listening = True
                        break
                except Exception as e:
                    pass
                
                if log_file_pattern and log_file_pattern.exists() and port_check_count > 0 and port_check_count % 10 == 0:
                    try:
                        with open(log_file_pattern, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            if lines:
                                last_lines = ''.join(lines[-5:])
                                if 'Exiting abnormally' in last_lines or 'Exiting.' in last_lines or 'LogExit: Exiting' in last_lines:
                                    scene_exited = True
                                    if USE_LOGGER:
                                        logger.error(f"Detected scene exit (from log file {log_file_pattern.name})")
                                    break
                    except:
                        pass
                
                port_check_count += 1
                if port_check_count % 5 == 0 and not USE_LOGGER:
                    print(f"Waiting for port {ports[0]} to listen... ({port_check_count * port_check_interval}s)")

                time.sleep(port_check_interval)
            
            if not port_listening:
                if scene_exited:
                    error_msg = f"Scene exited during startup (port {ports[0]} never started listening)"
                    if log_file_pattern and log_file_pattern.exists():
                        error_msg += f", check log file: {log_file_pattern}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                process_status = "unknown"
                try:
                    process_status = "port not listening; scene may have crashed or hung"
                except:
                    pass

                error_msg = f"Port {ports[0]} did not start listening within {max_port_checks * port_check_interval} seconds; scene may have crashed. {process_status}"
                if log_file_pattern and log_file_pattern.exists():
                    error_msg += f", check log file: {log_file_pattern}"
                logger.error(error_msg)
                logger.error("Check the scene log file for more details")
                raise Exception(error_msg)

            while waited < max_wait_time:
                try:
                    self.airsim_clients[index][0].ping()
                    objs = self.airsim_clients[index][0].simListSceneObjects()
                    if objs and len(objs) > 0:
                        if USE_LOGGER:
                            logger.info(f"Scene loaded. objects={len(objs)}")
                        ready = True
                        break
                    else:
                        if not USE_LOGGER and waited % 10 == 0:  # message10message
                            print(f"Connected but scene objects list is empty; waiting... {waited}/{max_wait_time}s")
                except Exception as e:
                    error_msg = str(e)
                    if not USE_LOGGER and waited % 10 == 0:  # message10message
                        if "ECONNREFUSED" in error_msg or "Connection refused" in error_msg:
                            print(f"Scene not ready; waiting... {waited}/{max_wait_time}s (connection refused)")
                        else:
                            print(f"Scene not ready; waiting... {waited}/{max_wait_time}s")
                waited += check_interval

            if not ready:
                logger.warning(f"Scene readiness check timed out ({max_wait_time}s); it may still be initializing.")
            
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('message')

        after = time.time()
        diff = after - before
        if USE_LOGGER:
            logger.info(f"message：{diff:.2f} message")
        else:
            print(f"message：{diff:.2f} message")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()



class TrajectoryExecutor:
    
    def __init__(self, 
                 scene_id="env_400",
                 sim_server_host="127.0.0.1",
                 sim_server_port=30000,  # message30000
                 gpu_id=0,
                 scene_index=1,
                 uav_vehicle_name="Drone_1",
                 target_object_name="UAV1",  # message，messageexecute_trajectorymessageUAV1-UAV20
                 target_asset_name=None,  # messageNone，messagetarget_object_namemessage
                 target_object_scale=(1.0, 1.0, 1.0),
                 uav_speed=5.0,
                 target_speed=3.0,
                 camera_name="0",
                 auto_start_scene=True,
                 pre_existing_client=None, 
                 pre_existing_sim_client_tool=None,  # message SimClientTool（message）
                 deterministic_step_mode=True):  # message AirSim message（message）
        self.scene_id = scene_id
        self.sim_server_host = sim_server_host
        self.sim_server_port = sim_server_port
        self.gpu_id = gpu_id
        self.scene_index = scene_index
        self.uav_vehicle_name = uav_vehicle_name
        self.target_object_name = target_object_name
        self.target_asset_name = target_asset_name if target_asset_name is not None else target_object_name
        self._target_asset_name_explicitly_set = (target_asset_name is not None)
        self.target_object_scale = target_object_scale
        self.uav_speed = uav_speed
        self.target_speed = target_speed
        self.camera_name = camera_name
        self.auto_start_scene = auto_start_scene
        self.pre_existing_client = pre_existing_client  # message
        self.pre_existing_sim_client_tool = pre_existing_sim_client_tool  # message SimClientTool
        self.deterministic_step_mode = bool(deterministic_step_mode)
        self._sim_paused_by_executor = False  # message

        
        self.client = None
        self.sim_client_tool = None
        self._connected_scene_id = None
        
        self._prev_frame_data = None
        
        self._abnormal_jumps = []  # message: [(frame_idx, jump_distance), ...]
    

    def _safe_sim_pause(self, pause: bool):
        try:
            self.client.simPause(bool(pause))
            self._sim_paused_by_executor = bool(pause)
        except Exception:
            pass

    def _safe_continue_for_frames(self, frames: int):
        if frames is None:
            return
        frames = int(frames)
        if frames <= 0:
            return
        try:
            self._safe_sim_pause(True)
            self.client.simContinueForFrames(frames)
            self._safe_sim_pause(True)
        except Exception:
            import time
            try:
                self._safe_sim_pause(False)
                time.sleep(0.02 * frames)
            finally:
                self._safe_sim_pause(True)

    def _get_vehicle_pos(self):
        pose = self.client.simGetVehiclePose(vehicle_name=self.uav_vehicle_name)
        p = pose.position
        return float(p.x_val), float(p.y_val), float(p.z_val)

    def _set_vehicle_pose_paused(self, x, y, z, quat, retries=3, tol_xy=0.3, tol_z=1.0):
        import time
        import numpy as np
        import airsim

        x, y, z = float(x), float(y), float(z)

        self._safe_sim_pause(True)
        last = None
        for k in range(int(retries)):
            pose = airsim.Pose(airsim.Vector3r(x, y, z), quat)
            rpc_ok = False
            last_rpc_err = None
            for rpc_try in range(5):
                try:
                    self.client.simSetVehiclePose(
                        pose,
                        ignore_collision=True,
                        vehicle_name=self.uav_vehicle_name,
                    )
                    rpc_ok = True
                    break
                except Exception as e:
                    last_rpc_err = e
                    err_l = str(e).lower()
                    if (
                        rpc_try < 4
                        and ("timeout" in err_l or "timed out" in err_l or "request timed out" in err_l)
                    ):
                        time.sleep(0.6 * (rpc_try + 1))
                        continue
                    raise
            if not rpc_ok and last_rpc_err is not None:
                raise last_rpc_err

            try:
                self.client.simContinueForFrames(2)
            except Exception:
                pass

            px, py, pz = self._get_vehicle_pos()
            last = np.array([px, py, pz], dtype=np.float32)

            err_xy = float(np.hypot(px - x, py - y))
            err_z = float(abs(pz - z))
            if (err_xy <= float(tol_xy)) and (err_z <= float(tol_z)):
                return True, last, float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))), err_xy, err_z

        err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))) if last is not None else float("inf")
        err_xy = float(np.hypot(last[0] - x, last[1] - y)) if last is not None else float("inf")
        err_z = float(abs(last[2] - z)) if last is not None else float("inf")
        return False, last, err, err_xy, err_z

    def _set_object_pose_paused(self, object_name, x, y, z, quat=None, retries=3, tol=1.0):
        import numpy as np
        import airsim

        x, y, z = float(x), float(y), float(z)
        if quat is None:
            quat = airsim.to_quaternion(0, 0, 0)

        self._safe_sim_pause(True)
        last = None
        for k in range(int(retries)):
            self.client.simSetObjectPose(
                object_name,
                airsim.Pose(airsim.Vector3r(x, y, z), quat)
            )
            pose = self.client.simGetObjectPose(object_name)
            if pose is None:
                continue
            p = pose.position
            last = np.array([float(p.x_val), float(p.y_val), float(p.z_val)], dtype=np.float32)
            if np.any(np.isnan(last)):
                continue
            err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32)))
            if err <= float(tol):
                return True, last, err

        err = float(np.linalg.norm(last - np.array([x, y, z], dtype=np.float32))) if last is not None and not np.any(np.isnan(last)) else float("inf")
        return False, last, err

    def _step_if_needed(self, frames: int = 1):
        if not getattr(self, "deterministic_step_mode", False):
            return
        self._safe_continue_for_frames(frames)

    def connect(self, reuse_connection=True, max_retries=3, retry_delay=2):
        import time
        
        main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
        
        if reuse_connection and self.client is not None and self._connected_scene_id == main_scene_id:
            try:
                self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                if getattr(self, "deterministic_step_mode", False):
                    self._safe_sim_pause(True)
                return self.client
            except:
                print(f"⚠ message，message...")
                self.client = None
                self.sim_client_tool = None
                self._connected_scene_id = None
        
        if self.pre_existing_client is not None:
            self.client = self.pre_existing_client
            if self.pre_existing_sim_client_tool is not None:
                self.sim_client_tool = self.pre_existing_sim_client_tool
            self._connected_scene_id = main_scene_id
            return self.client
        
        if not self.auto_start_scene or not MSGPACKRPC_AVAILABLE:
            raise RuntimeError("message AirSim（message auto_start_scene message msgpackrpc）")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                if isinstance(self.scene_id, list) and isinstance(self.gpu_id, list):
                    if len(self.scene_id) != len(self.gpu_id):
                        raise RuntimeError(f"message({len(self.scene_id)})messageGPUmessage({len(self.gpu_id)})message")
                    open_scenes = self.scene_id
                    gpus = self.gpu_id
                    main_scene_id = open_scenes[0]
                else:
                    open_scenes = [self.scene_id] if not isinstance(self.scene_id, list) else self.scene_id
                    gpus = [self.gpu_id] if not isinstance(self.gpu_id, list) else self.gpu_id
                    main_scene_id = self.scene_id if not isinstance(self.scene_id, list) else self.scene_id[0]
                
                machines_info = [{
                    'MACHINE_IP': self.sim_server_host,
                    'SOCKET_PORT': self.sim_server_port,
                    'open_scenes': open_scenes,
                    'gpus': gpus
                }]
                
                if self.sim_client_tool is not None:
                    try:
                        self.sim_client_tool._closeConnection()
                        self.sim_client_tool._closeSocketConnection()
                    except:
                        pass
                
                if os.environ.get("DAGGER_MULTI_WORKER") != "1":
                    try:
                        tmp_client = msgpackrpc.Client(
                            msgpackrpc.Address(self.sim_server_host, self.sim_server_port),
                            timeout=30,
                        )
                        try:
                            tmp_client.call('close_scenes', self.sim_server_host)
                        except Exception:
                            pass
                        try:
                            tmp_client.close()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    import time as _time
                    _time.sleep(3)
                self.sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                self.sim_client_tool.run_call()
                
                if len(self.sim_client_tool.airsim_clients) > 0 and len(self.sim_client_tool.airsim_clients[0]) > 0:
                    if isinstance(self.scene_id, list):
                        scene_index = open_scenes.index(main_scene_id)
                        self.client = self.sim_client_tool.airsim_clients[0][scene_index]
                    else:
                        self.client = self.sim_client_tool.airsim_clients[0][0]
                    
                    if self.client is None:
                        raise RuntimeError("AirSim message None")
                    
                    vehicle_names = self.client.listVehicles()
                    if self.uav_vehicle_name not in vehicle_names:
                        raise RuntimeError(f"message '{self.uav_vehicle_name}'，message: {vehicle_names}")
                    
                    self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                    max_retries = 5
                    retry_delay = 2
                    for attempt in range(max_retries):
                        try:
                            self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                error_msg = str(e)
                                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                    time.sleep(retry_delay)
                                    continue
                            raise
                    
                    print(f"✓ message AirSim（message: {main_scene_id}）")
                    self.client._sim_client_tool = self.sim_client_tool
                    self._connected_scene_id = main_scene_id
                    
                    return self.client
                else:
                    raise RuntimeError("message AirSim message")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"⚠ message（message {attempt + 1}/{max_retries}）: {e}")
                    print(f"  message {retry_delay} message...")
                    self.client = None
                    if self.sim_client_tool is not None:
                        try:
                            self.sim_client_tool._closeConnection()
                            self.sim_client_tool._closeSocketConnection()
                        except:
                            pass
                        self.sim_client_tool = None
                else:
                    raise RuntimeError(f"message（message {max_retries} message）: {e}")
        
        raise RuntimeError(f"message（message {max_retries} message）: {last_error}")
    
    def disconnect(self):
        self.client = None
        self.sim_client_tool = None
        self._connected_scene_id = None
    
    def load_trajectory(self, json_path):
        json_path = Path(json_path)
        
        if json_path.name.endswith('_uav.json') or json_path.name.endswith('_target.json'):
            base_name = json_path.name.replace('_uav.json', '').replace('_target.json', '')
            uav_file = json_path.parent / f"{base_name}_uav.json"
            target_file = json_path.parent / f"{base_name}_target.json"
            
            if not uav_file.exists():
                raise FileNotFoundError(f"UAVmessage: {uav_file}")
            with open(uav_file, 'r', encoding='utf-8') as f:
                uav_data = json.load(f)
            if 'uav_trajectory' not in uav_data:
                raise ValueError(f"UAVmessage：message 'uav_trajectory' message")
            uav_traj_data = uav_data['uav_trajectory']
            
            if not target_file.exists():
                raise FileNotFoundError(f"message: {target_file}")
            with open(target_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            if 'target_trajectory' not in target_data:
                raise ValueError(f"message：message 'target_trajectory' message")
            target_traj_data = target_data['target_trajectory']
            
            is_dataset_format = False
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'trajectory' in data and isinstance(data['trajectory'], list):
                uav_traj_data = []
                for frame in data['trajectory']:
                    if 'uav_position' in frame and frame['uav_position'] is not None:
                        uav_pos = frame['uav_position']
                        if isinstance(uav_pos, dict) and 'x' in uav_pos and 'y' in uav_pos and 'z' in uav_pos:
                            uav_traj_data.append([uav_pos['x'], uav_pos['y'], uav_pos['z']])
                target_file = json_path.parent / "target_trajectory.json"
                if target_file.exists():
                    with open(target_file, 'r', encoding='utf-8') as f:
                        target_data = json.load(f)
                    if 'target_trajectory_airsim' in target_data and isinstance(target_data['target_trajectory_airsim'], list):
                        target_traj_data = []
                        for p in target_data['target_trajectory_airsim']:
                            if isinstance(p, dict) and 'x' in p and 'y' in p and 'z' in p:
                                target_traj_data.append([p['x'], p['y'], p['z']])
                    else:
                        target_traj_data = []
                else:
                    target_traj_data = []
                if not target_traj_data:
                    for frame in data['trajectory']:
                        if 'target_position' in frame and frame['target_position'] is not None:
                            target_pos = frame['target_position']
                            if isinstance(target_pos, dict) and 'x' in target_pos and 'y' in target_pos and 'z' in target_pos:
                                target_traj_data.append([target_pos['x'], target_pos['y'], target_pos['z']])
                is_dataset_format = True
            elif 'uav_trajectory' in data and 'target_trajectory' in data:
                uav_traj_data = data['uav_trajectory']
                target_traj_data = data['target_trajectory']
                is_dataset_format = False
            else:
                raise ValueError(f"message：message 'uav_trajectory'/'target_trajectory' message 'trajectory' message")
        
        if not isinstance(uav_traj_data, list) or len(uav_traj_data) == 0:
            raise ValueError(f"message：message UAV message（message {len(uav_traj_data) if isinstance(uav_traj_data, list) else 0} message）。message 'uav_position' message。")
        if not isinstance(target_traj_data, list) or len(target_traj_data) == 0:
            raise ValueError(f"message：message（message {len(target_traj_data) if isinstance(target_traj_data, list) else 0} message）。message 'target_position' message。")
        
        uav_traj = np.array(uav_traj_data)
        target_traj = np.array(target_traj_data)
        
        if uav_traj.ndim == 0:
            raise ValueError(f"UAVmessage：message2message，message0message")
        elif uav_traj.ndim == 1:
            uav_traj = uav_traj.reshape(1, -1)
        
        if target_traj.ndim == 0:
            raise ValueError(f"message：message2message，message0message")
        elif target_traj.ndim == 1:
            target_traj = target_traj.reshape(1, -1)
        
        uav_traj_airsim = np.zeros_like(uav_traj)
        target_traj_airsim = np.zeros_like(target_traj)
        
        if is_dataset_format:
            uav_traj_airsim[:, 0] = uav_traj[:, 0]  # xmessage
            uav_traj_airsim[:, 1] = uav_traj[:, 1]  # ymessage（message）
            uav_traj_airsim[:, 2] = -uav_traj[:, 2]  # zmessage（message，message）
            
            target_traj_airsim[:, 0] = target_traj[:, 0]  # xmessage
            target_traj_airsim[:, 1] = target_traj[:, 1]  # ymessage（message）
            target_traj_airsim[:, 2] = -target_traj[:, 2]  # zmessage（message，message）
        else:
            uav_traj_airsim[:, 0] = uav_traj[:, 0]  # xmessage
            uav_traj_airsim[:, 1] = -uav_traj[:, 1]  # ymessage
            uav_traj_airsim[:, 2] = -uav_traj[:, 2]  # zmessage（NEDmessage）
            
            target_traj_airsim[:, 0] = target_traj[:, 0]  # xmessage
            target_traj_airsim[:, 1] = -target_traj[:, 1]  # ymessage
            target_traj_airsim[:, 2] = -target_traj[:, 2]  # zmessage（NEDmessage）
        
        return uav_traj_airsim, target_traj_airsim
    
    def _safe_call_airsim(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (StreamClosedError, OSError, ConnectionError) as e:
            return None
        except Exception as e:
            error_msg = str(e).lower()
            if "stream is closed" in error_msg or "connection" in error_msg or "streamclosederror" in error_msg:
                return None
            raise
    
    def reset_collision_info(self):
        try:
            collision_info = self._safe_call_airsim(self.client.simGetCollisionInfo, vehicle_name=self.uav_vehicle_name)
            collision_info = self._safe_call_airsim(self.client.simGetCollisionInfo, vehicle_name=self.uav_vehicle_name)
            self._safe_call_airsim(self.client.simContinueForFrames, 1)
        except Exception as e:
            pass
    

    def teleport_to_start(self, x, y, z, target_x=None, target_y=None, target_z=None, quaternion=None):
        import time
        import numpy as np
        import airsim

        max_teleport_retries = 3

        if quaternion is not None:
            try:
                quat = airsim.Quaternionr(
                    w_val=float(quaternion[0]),
                    x_val=float(quaternion[1]),
                    y_val=float(quaternion[2]),
                    z_val=float(quaternion[3]),
                )
            except Exception:
                quat = airsim.to_quaternion(0, 0, 0)
        else:
            if target_x is not None and target_y is not None:
                yaw = float(np.arctan2(float(target_y) - float(y), float(target_x) - float(x)))
            else:
                yaw = 0.0
            quat = airsim.to_quaternion(0, 0, yaw)

        last = None
        last_err = None

        for attempt in range(max_teleport_retries):
            try:
                try:
                    self.client.enableApiControl(False, vehicle_name=self.uav_vehicle_name)
                except Exception:
                    pass
                try:
                    self.client.armDisarm(False, vehicle_name=self.uav_vehicle_name)
                except Exception:
                    pass

                ok, last, err, err_xy, err_z = self._set_vehicle_pose_paused(
                    x, y, z, quat,
                    retries=1,          # message attempt message set message，message
                    tol_xy=0.3,
                    tol_z=1.0
                )
                last_err = (err, err_xy, err_z)

                if not ok:
                    if attempt < max_teleport_retries - 1:
                        time.sleep(0.1)
                        continue
                    raise RuntimeError(
                        f"UAVmessage（message {max_teleport_retries} message）："
                        f"message({float(x):.2f}, {float(y):.2f}, {float(z):.2f})，"
                        f"message({last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f})，"
                        f"message{err:.2f}m (XY:{err_xy:.2f}m, Z:{err_z:.2f}m)"
                    )

                max_retries = 5
                retry_delay = 1.0
                for k in range(max_retries):
                    try:
                        self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                        break
                    except Exception as e:
                        if k < max_retries - 1:
                            msg = str(e).lower()
                            if "timeout" in msg or "timed out" in msg:
                                time.sleep(retry_delay)
                                continue
                        raise

                if getattr(self, "deterministic_step_mode", False):
                    self._step_if_needed(1)
                else:
                    self._safe_sim_pause(False)
                    time.sleep(0.02)

                self.reset_collision_info()

                return  # message

            except Exception:
                if attempt < max_teleport_retries - 1:
                    time.sleep(0.2)
                    continue
                raise

    def spawn_target_object(self, x, y, z):
        try:
            try:
                self.client.simDestroyObject(self.target_object_name)
                try:
                    self.client.simContinueForFrames(1)
                except:
                    pass
            except:
                try:
                    pattern = self.target_object_name + ".*"
                    existing_objects = self.client.simListSceneObjects(pattern)
                    for obj_name in existing_objects:
                        try:
                            self.client.simDestroyObject(obj_name)
                        except:
                            pass
                    try:
                        self.client.simContinueForFrames(1)
                    except:
                        pass
                except:
                    pass
            
            pose = airsim.Pose(
                airsim.Vector3r(x, y, z),
                airsim.to_quaternion(0, 0, 0)  # message
            )
            
            scale_vector = airsim.Vector3r(self.target_object_scale[0], self.target_object_scale[1], self.target_object_scale[2])
            
            success = self.client.simSpawnObject(
                self.target_object_name,
                self.target_asset_name,
                pose,
                scale_vector,
                physics_enabled=False,  # message
                is_blueprint=False
            )
            
            if success:
                try:
                    self.client.simContinueForFrames(1)
                except:
                    pass
                
                max_verify_attempts = 5
                for verify_attempt in range(max_verify_attempts):
                    try:
                        verify_pose = self.client.simGetObjectPose(self.target_object_name)
                        if verify_pose is not None:
                            actual_pos = np.array([
                                verify_pose.position.x_val,
                                verify_pose.position.y_val,
                                verify_pose.position.z_val
                            ])
                            
                            if np.any(np.isnan(actual_pos)):
                                if verify_attempt < max_verify_attempts - 1:
                                    print(f"  ⚠ message NaN，message ({verify_attempt + 1}/{max_verify_attempts})...")
                                    try:
                                        self.client.simContinueForFrames(1)
                                    except:
                                        pass
                                    continue
                                else:
                                    print(f"✗ message NaN，message")
                                    try:
                                        all_objects = self.client.simListSceneObjects(".*")
                                        matching_objects = [obj for obj in all_objects if self.target_object_name.lower() in obj.lower()]
                                        if matching_objects:
                                            print(f"  message: {matching_objects[:5]}")
                                        print(f"  message: {len(all_objects)}")
                                    except:
                                        pass
                                    return False
                            
                            return True
                        else:
                            if verify_attempt < max_verify_attempts - 1:
                                safe_log(f"⚠ message，message ({verify_attempt + 1}/{max_verify_attempts})...", scene_id=self.scene_id)
                                try:
                                    self.client.simContinueForFrames(1)
                                except:
                                    pass
                                continue
                            else:
                                safe_log(f"⚠ message：message", scene_id=self.scene_id)
                                return False
                    except Exception as e:
                        if verify_attempt < max_verify_attempts - 1:
                            safe_log(f"⚠ message，message ({verify_attempt + 1}/{max_verify_attempts}): {e}", scene_id=self.scene_id)
                            continue
                        else:
                            safe_log(f"✗ message: {e}", scene_id=self.scene_id)
                            return False
                
                print(f"⚠ message：message")
                return False
            else:
                print(f"✗ simSpawnObject message")
                return False
                
        except Exception as e:
            print(f"✗ message: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    def teleport_object_to_start(self, x, y, z):
        import time
        import numpy as np
        import airsim

        try:
            if not self.spawn_target_object(x, y, z):
                return False

            pose = self.client.simGetObjectPose(self.target_object_name)
            if pose is not None:
                p = pose.position
                cur = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)
                err = float(np.linalg.norm(cur - np.array([float(x), float(y), float(z)], dtype=np.float32)))
            else:
                err = float("inf")

            if err > 1.0:
                ok, last, err2 = self._set_object_pose_paused(
                    self.target_object_name, x, y, z,
                    quat=airsim.to_quaternion(0, 0, 0),
                    retries=3,
                    tol=1.0
                )
                if not ok:
                    print(
                        f"✗ message：message({float(x):.2f},{float(y):.2f},{float(z):.2f})，"
                        f"message({last[0]:.2f},{last[1]:.2f},{last[2]:.2f})，message{err2:.2f}m"
                    )
                    return False

            self._step_if_needed(1)
            return True

        except Exception as e:
            print(f"✗ message: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_object_position(self):
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                pose = self.client.simGetObjectPose(self.target_object_name)
                if pose is None:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
                if np.any(np.isnan(pos)):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
                return pos
            except (TimeoutError, Exception) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None
    
    def get_object_pose(self):
        try:
            pose = self.client.simGetObjectPose(self.target_object_name)
            if pose is None:
                return None, None
            position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
            if np.any(np.isnan(position)):
                return None, None
            orientation = np.array([pose.orientation.w_val, pose.orientation.x_val, 
                                    pose.orientation.y_val, pose.orientation.z_val])
            return position, orientation
        except Exception as e:
            return None, None
    
    def get_uav_state(self):
        state = self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.uav_vehicle_name)
        has_collided = collision_info.has_collided if collision_info else False
        collision_time_stamp = None
        if collision_info and has_collided:
            try:
                if hasattr(collision_info, 'time_stamp'):
                    collision_time_stamp = collision_info.time_stamp
                elif hasattr(collision_info, 'time_stamp_ns'):
                    collision_time_stamp = collision_info.time_stamp_ns
            except:
                pass
        
        return {
            'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
            'orientation': np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val]),
            'has_collided': has_collided,
            'collision_time_stamp': collision_time_stamp
        }
    
    def get_camera_images(self):
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                responses_rgb = self.client.simGetImages([
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
                ], vehicle_name=self.uav_vehicle_name)
                
                rgb_response = responses_rgb[0]
                rgb_img = None
                if rgb_response.image_data_uint8:
                    rgb_img = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
                    rgb_img = rgb_img.reshape(rgb_response.height, rgb_response.width, 3)
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # AirSimmessageBGR，messageRGB
                
                depth_img = None
                return rgb_img, depth_img
            except (TimeoutError, Exception) as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    safe_log(f"⚠ message（message {attempt + 1}/{max_retries}）: {error_msg}", scene_id=self.scene_id)
                    time.sleep(retry_delay)
                    continue
                else:
                    safe_log(f"✗ message（message {max_retries} message）: {error_msg}", scene_id=self.scene_id)
                    import traceback
                    traceback.print_exc()
                    return None, None
        
        return None, None
    

    def move_target_object(self, target_pos):
        import airsim
        import numpy as np

        try:
            test_pose = self.client.simGetObjectPose(self.target_object_name)
            if test_pose is None:
                if self.target_asset_name is None:
                    self.target_asset_name = self.target_object_name
                if not self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                    safe_log(
                        f"⚠ message {self.target_object_name}，message",
                        scene_id=self.scene_id
                    )
            else:
                test_pos = np.array([test_pose.position.x_val, test_pose.position.y_val, test_pose.position.z_val])
                if np.any(np.isnan(test_pos)):
                    if self.target_asset_name is None:
                        self.target_asset_name = self.target_object_name
                    if not self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])):
                        safe_log(
                            f"⚠ message {self.target_object_name} message NaN，message",
                            scene_id=self.scene_id
                        )
        except Exception as e:
            if self.target_asset_name is None:
                self.target_asset_name = self.target_object_name
            try:
                self.spawn_target_object(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            except Exception as spawn_error:
                safe_log(
                    f"⚠ message {self.target_object_name}: {e}, message: {spawn_error}",
                    scene_id=self.scene_id
                )

        pose_quat = airsim.to_quaternion(0, 0, 0)
        ok, last, err = self._set_object_pose_paused(
            self.target_object_name,
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
            quat=pose_quat,
            retries=2,
            tol=1.0
        )
        if not ok:
            if last is not None:
                safe_log(
                    f"⚠ message：message({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f}) "
                    f"message({last[0]:.2f},{last[1]:.2f},{last[2]:.2f}) err={err:.2f}m",
                    scene_id=self.scene_id
                )
            else:
                safe_log(
                    f"⚠ message：message({float(target_pos[0]):.2f},{float(target_pos[1]):.2f},{float(target_pos[2]):.2f}) "
                    f"message（message）",
                    scene_id=self.scene_id
                )

    def cleanup_old_frames(self, dataset_dir):
        dataset_path = Path(dataset_dir)
        rgb_dir = dataset_path / "rgb"
        
        if rgb_dir.exists():
            for old_file in rgb_dir.glob("frame_*.png"):
                try:
                    old_file.unlink()
                except Exception:
                    pass  # message
    
    def save_frame_data(self, frame_idx, rgb_img, depth_img, dataset_dir):
        dataset_path = Path(dataset_dir)
        
        rgb_dir = dataset_path / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        
        if rgb_img is not None:
            rgb_path = rgb_dir / f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    def _prepare_target_object(self):
        if self._target_asset_name_explicitly_set:
            selected_uav_name = self.target_asset_name
        else:
            selected_uav_num = random.randint(1, 20)
            selected_uav_name = f"UAV{selected_uav_num}"
            self.target_asset_name = selected_uav_name
        
        import time as time_module
        unique_suffix = int(time_module.time() * 1000) % 100000  # message5message
        random_suffix = random.randint(1000, 9999)  # message，message
        unique_object_name = f"{selected_uav_name}_{unique_suffix}_{random_suffix}"
        
        self.target_object_name = unique_object_name
        
        return selected_uav_name
    
    def _prepare_dataset_directory(self, trajectory_name, dataset_base_dir, save_dataset):
        dataset_dir = None
        if save_dataset:
            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            dataset_dir = str(dataset_path)
            self.cleanup_old_frames(dataset_dir)
        return dataset_dir
    
    def _initialize_simulation(self, uav_traj, target_traj):
        self.connect()
        max_retries = 10  # message
        base_retry_delay = 2  # message
        scene_restart_threshold = 1  # message1message（message）
        last_error = None
        scene_restarted = False
        consecutive_timeouts = 0  # message
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    try:
                        self.client.getMultirotorState(vehicle_name=self.uav_vehicle_name)
                        consecutive_timeouts = 0  # message，message
                    except Exception as conn_e:
                        consecutive_timeouts += 1
                        safe_log(f"⚠ [{self.scene_id}] message，message: {str(conn_e)}", scene_id=self.scene_id)
                        if "timeout" in str(conn_e).lower() or "timed out" in str(conn_e).lower():
                            sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                            if not scene_restarted and sim_tool_to_use is not None:
                                safe_log(f"🔄 [{self.scene_id}] message，message...", scene_id=self.scene_id)
                                try:
                                    try:
                                        if hasattr(sim_tool_to_use, '_closeConnection'):
                                            sim_tool_to_use._closeConnection()
                                    except:
                                        pass
                                    try:
                                        if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                            sim_tool_to_use._closeSocketConnection()
                                    except:
                                        pass
                                    
                                    time.sleep(3)
                                    
                                    main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                                    machines_info = [{
                                        'MACHINE_IP': self.sim_server_host,
                                        'SOCKET_PORT': self.sim_server_port,
                                        'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                                        'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                                    }]
                                    if os.environ.get("DAGGER_MULTI_WORKER") != "1":
                                        try:
                                            tmp_client = msgpackrpc.Client(
                                                msgpackrpc.Address(self.sim_server_host, self.sim_server_port),
                                                timeout=30,
                                            )
                                            try:
                                                tmp_client.call('close_scenes', self.sim_server_host)
                                            except Exception:
                                                pass
                                            try:
                                                tmp_client.close()
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                    new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                                    new_sim_client_tool.run_call()
                                    
                                    if self.pre_existing_sim_client_tool is not None:
                                        self.pre_existing_sim_client_tool = new_sim_client_tool
                                    self.sim_client_tool = new_sim_client_tool
                                    
                                    scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                                    if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                                        self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                                    else:
                                        self.client = new_sim_client_tool.airsim_clients[0][0]
                                    
                                    if self.pre_existing_client is not None:
                                        self.pre_existing_client = self.client
                                    
                                    self._connected_scene_id = main_scene_id
                                    scene_restarted = True
                                    safe_log(f"✓ [{self.scene_id}] message，message...", scene_id=self.scene_id)
                                    time.sleep(5)  # message
                                    consecutive_timeouts = 0  # message
                                except Exception as restart_e:
                                    safe_log(f"⚠ [{self.scene_id}] message: {str(restart_e)}，message...", scene_id=self.scene_id)
                                    scene_restarted = True  # message，message
                        else:
                            try:
                                self.connect(reuse_connection=False)
                            except:
                                pass
                
                sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                if attempt >= scene_restart_threshold and not scene_restarted and sim_tool_to_use is not None:
                    safe_log(f"🔄 [{self.scene_id}] enableApiControl message{attempt}message，message...", scene_id=self.scene_id)
                    try:
                        try:
                            if hasattr(sim_tool_to_use, '_closeConnection'):
                                sim_tool_to_use._closeConnection()
                        except:
                            pass
                        try:
                            if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                sim_tool_to_use._closeSocketConnection()
                        except:
                            pass
                        
                        time.sleep(3)
                        
                        main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                        machines_info = [{
                            'MACHINE_IP': self.sim_server_host,
                            'SOCKET_PORT': self.sim_server_port,
                            'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                            'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                        }]
                        new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                        new_sim_client_tool.run_call()
                        
                        if self.pre_existing_sim_client_tool is not None:
                            self.pre_existing_sim_client_tool = new_sim_client_tool
                        self.sim_client_tool = new_sim_client_tool
                        
                        scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                        if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                            self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                        else:
                            self.client = new_sim_client_tool.airsim_clients[0][0]
                        
                        if self.pre_existing_client is not None:
                            self.pre_existing_client = self.client
                        
                        self._connected_scene_id = main_scene_id
                        scene_restarted = True
                        safe_log(f"✓ [{self.scene_id}] message，message...", scene_id=self.scene_id)
                        time.sleep(5)  # message
                    except Exception as restart_e:
                        safe_log(f"⚠ [{self.scene_id}] message: {str(restart_e)}，message...", scene_id=self.scene_id)
                        scene_restarted = True  # message，message
                
                self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                if attempt > 0:
                    if scene_restarted:
                        safe_log(f"✓ [{self.scene_id}] enableApiControl message（message{attempt+1}message）", scene_id=self.scene_id)
                    else:
                        safe_log(f"✓ [{self.scene_id}] enableApiControl message（message{attempt+1}message）", scene_id=self.scene_id)
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    
                    if is_timeout:
                        consecutive_timeouts += 1
                        sim_tool_to_use = self.pre_existing_sim_client_tool if self.pre_existing_sim_client_tool is not None else self.sim_client_tool
                        if consecutive_timeouts >= 1 and not scene_restarted and sim_tool_to_use is not None:
                            safe_log(f"🔄 [{self.scene_id}] message{consecutive_timeouts}message，message...", scene_id=self.scene_id)
                            try:
                                try:
                                    if hasattr(sim_tool_to_use, '_closeConnection'):
                                        sim_tool_to_use._closeConnection()
                                except:
                                    pass
                                try:
                                    if hasattr(sim_tool_to_use, '_closeSocketConnection'):
                                        sim_tool_to_use._closeSocketConnection()
                                except:
                                    pass
                                
                                time.sleep(3)
                                
                                main_scene_id = self.scene_id if isinstance(self.scene_id, str) else self.scene_id[0]
                                machines_info = [{
                                    'MACHINE_IP': self.sim_server_host,
                                    'SOCKET_PORT': self.sim_server_port,
                                    'open_scenes': [main_scene_id] if isinstance(self.scene_id, str) else self.scene_id,
                                    'gpus': [self.gpu_id] if isinstance(self.gpu_id, (int, str)) else self.gpu_id
                                }]
                                new_sim_client_tool = AirVLNSimulatorClientTool(machines_info)
                                new_sim_client_tool.run_call()
                                
                                if self.pre_existing_sim_client_tool is not None:
                                    self.pre_existing_sim_client_tool = new_sim_client_tool
                                self.sim_client_tool = new_sim_client_tool
                                
                                scene_index = self.scene_index if isinstance(self.scene_id, str) else 0
                                if len(new_sim_client_tool.airsim_clients) > 0 and len(new_sim_client_tool.airsim_clients[0]) > scene_index:
                                    self.client = new_sim_client_tool.airsim_clients[0][scene_index]
                                else:
                                    self.client = new_sim_client_tool.airsim_clients[0][0]
                                
                                if self.pre_existing_client is not None:
                                    self.pre_existing_client = self.client
                                
                                self._connected_scene_id = main_scene_id
                                scene_restarted = True
                                consecutive_timeouts = 0  # message
                                safe_log(f"✓ [{self.scene_id}] message，message...", scene_id=self.scene_id)
                                time.sleep(5)  # message
                            except Exception as restart_e:
                                safe_log(f"⚠ [{self.scene_id}] message: {str(restart_e)}，message...", scene_id=self.scene_id)
                                scene_restarted = True  # message，message
                    else:
                        consecutive_timeouts = 0
                    
                    retry_delay = base_retry_delay * (attempt + 1)
                    if is_timeout:
                        safe_log(f"⚠ [{self.scene_id}] enableApiControl message，{retry_delay}message ({attempt+1}/{max_retries})", scene_id=self.scene_id)
                        time.sleep(retry_delay)
                        continue
                    else:
                        safe_log(f"⚠ [{self.scene_id}] enableApiControl message: {error_msg[:100]}，{retry_delay}message ({attempt+1}/{max_retries})", scene_id=self.scene_id)
                        time.sleep(retry_delay)
                        continue
                else:
                    error_msg = str(last_error)
                    safe_log(f"❌ [{self.scene_id}] enableApiControl message（message{max_retries}message）: {error_msg[:200]}", scene_id=self.scene_id)
                    raise
        self.client.armDisarm(True, vehicle_name=self.uav_vehicle_name)
        
        u0 = uav_traj[0]
        t0 = target_traj[0]
        
        self.teleport_object_to_start(t0[0], t0[1], t0[2])
        
        self.teleport_to_start(u0[0], u0[1], u0[2], 
                              target_x=t0[0], target_y=t0[1], target_z=t0[2])
        
        try:
            max_retries = 3
            pos_error = float('inf')
            cur_pos = None
            
            for retry in range(max_retries):
                uav_state = self.get_uav_state()
                cur_pos = uav_state['position']
                
                pos_error = np.linalg.norm(cur_pos - np.array(u0))
                delta_xy = cur_pos[:2] - u0[:2]
                pos_error_xy = np.linalg.norm(delta_xy)
                pos_error_z = abs(cur_pos[2] - u0[2])
                
                if pos_error <= 1.0:
                    break  # message，message
                
                if retry < max_retries - 1:  # message
                    print(
                        f"⚠ message：message（message {retry + 1}/{max_retries}）："
                        f"message ({u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f})，"
                        f"message ({cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f})，"
                        f"message {pos_error:.2f} m (XY: {pos_error_xy:.2f}m, Z: {pos_error_z:.2f}m)"
                    )
                    self.teleport_to_start(
                        u0[0], u0[1], u0[2],
                        target_x=t0[0], target_y=t0[1], target_z=t0[2]
                    )
                    try:
                        self.client.simContinueForFrames(5)  # messageteleport_to_startmessage，message5message
                    except:
                        pass
                    time.sleep(0.1)  # messageteleport_to_startmessage，message0.1message
            
            if pos_error > 1.0 and cur_pos is not None:
                final_pos_error_xy = np.linalg.norm(cur_pos[:2] - u0[:2])
                final_pos_error_z = abs(cur_pos[2] - u0[2])
                print(
                    f"⚠ message：message（message）："
                    f"message ({u0[0]:.2f}, {u0[1]:.2f}, {u0[2]:.2f})，"
                    f"message ({cur_pos[0]:.2f}, {cur_pos[1]:.2f}, {cur_pos[2]:.2f})，"
                    f"message {pos_error:.2f} m (XY: {final_pos_error_xy:.2f}m, Z: {final_pos_error_z:.2f}m)"
                )
            
            dx = t0[0] - cur_pos[0]
            dy = t0[1] - cur_pos[1]
            yaw = np.arctan2(dy, dx)  # message
            
            quat = airsim.to_quaternion(0, 0, yaw)
            
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(cur_pos[0], cur_pos[1], cur_pos[2]), quat),
                ignore_collision=True,
                vehicle_name=self.uav_vehicle_name
            )
            
            try:
                self.client.simContinueForFrames(1)
            except:
                pass
        except Exception as e:
            print(f"⚠ message：message: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_collision_state(self):
        self.reset_collision_info()
    
    def _ensure_uav_flying_state(self):
        try:
            uav_state = self.get_uav_state()
            cur_pos = uav_state['position']
            
            pos2_now = self.get_object_position()
            if pos2_now is not None:
                dx = pos2_now[0] - cur_pos[0]
                dy = pos2_now[1] - cur_pos[1]
                yaw = np.arctan2(dy, dx)  # message
            else:
                orientation = uav_state['orientation']
                yaw = 0.0  # message
            
            quat = airsim.to_quaternion(0, 0, yaw)
            
            self.client.simSetVehiclePose(
                airsim.Pose(airsim.Vector3r(cur_pos[0], cur_pos[1], cur_pos[2]), quat),
                ignore_collision=True,
                vehicle_name=self.uav_vehicle_name
            )
        except:
            pass
    
    def _quaternion_to_euler(self, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        euler = rotation.as_euler('xyz', degrees=False)
        return {
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2])
        }
    
    def _world_to_body_frame(self, vector_world, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        vector_body = rotation.inv().apply(vector_world)
        vector_body[2] = -vector_body[2]
        return vector_body

    def _airsim_to_body_frame(self, vector_world, quat_w, quat_x, quat_y, quat_z):
        rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w])
        vector_body = rotation.inv().apply(vector_world)
        return vector_body
    
    def _append_trajectory_data(self, frame_idx, uav_state, cur1_pos, pos2_now,
                                merged_trajectory_data, next_target_pos_airsim=None):
        uav_pos = np.array([
            float(cur1_pos[0]),
            float(cur1_pos[1]),
            float(-cur1_pos[2])
        ])
        
        uav_quat = uav_state['orientation']
        uav_quat_w = float(uav_quat[0])
        uav_quat_x = float(uav_quat[1])
        uav_quat_y = float(uav_quat[2])
        uav_quat_z = float(uav_quat[3])
        
        uav_euler = self._quaternion_to_euler(uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z)
        
        if self._prev_frame_data is not None and 'frame_data' in self._prev_frame_data:
            prev_frame_data = self._prev_frame_data['frame_data']
            prev_pos = self._prev_frame_data['uav_position']
            prev_quat = self._prev_frame_data['uav_orientation_quaternion']
            prev_euler = self._prev_frame_data['uav_orientation_euler']
            
            position_diff_world = uav_pos - np.array([prev_pos['x'], prev_pos['y'], prev_pos['z']])
            
            velocity_body = self._world_to_body_frame(
                position_diff_world,
                prev_quat['w'], prev_quat['x'], prev_quat['y'], prev_quat['z']
            )
            
            prev_yaw = prev_euler['yaw']  # message
            current_yaw = uav_euler['yaw']  # message
            yaw_diff = current_yaw - prev_yaw
            
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            
            yaw_rate_deg = np.degrees(yaw_diff)
            
            prev_frame_data["velocity_in_body_frame"] = {
                "x": float(velocity_body[0]),
                "y": float(velocity_body[1]),
                "z": float(velocity_body[2])
            }
            prev_frame_data["yaw_rate"] = float(yaw_rate_deg)
        
        frame_data = {
            "frame_idx": frame_idx,
            "uav_position": {
                "x": uav_pos[0],
                "y": uav_pos[1],
                "z": uav_pos[2]
            },
            "uav_orientation_quaternion": {
                "w": uav_quat_w,
                "x": uav_quat_x,
                "y": uav_quat_y,
                "z": uav_quat_z
            },
            "uav_orientation_euler": uav_euler
        }
        
        if next_target_pos_airsim is not None:
            next_target_pos_world = np.array([
                float(next_target_pos_airsim[0]),
                float(next_target_pos_airsim[1]),
                float(-next_target_pos_airsim[2])
            ])
            
            relative_position = next_target_pos_world - uav_pos
            
            relative_position_body = self._world_to_body_frame(
                relative_position,
                uav_quat_w, uav_quat_x, uav_quat_y, uav_quat_z
            )
            
            frame_data["target_position"] = {
                "x": next_target_pos_world[0],
                "y": next_target_pos_world[1],
                "z": next_target_pos_world[2]
            }
            frame_data["relative_position"] = {
                "x": float(relative_position[0]),
                "y": float(relative_position[1]),
                "z": float(relative_position[2])
            }
            frame_data["target_position_in_body_frame"] = {
                "x": float(relative_position_body[0]),
                "y": float(relative_position_body[1]),
                "z": float(relative_position_body[2])
            }
            distance = np.linalg.norm(relative_position)
            frame_data["distance"] = float(distance)
        else:
            frame_data["target_position"] = None
            frame_data["relative_position"] = None
            frame_data["target_position_in_body_frame"] = None
            frame_data["distance"] = None
        
        frame_data["velocity_in_body_frame"] = None
        frame_data["yaw_rate"] = None
        
        self._prev_frame_data = {
            "frame_data": frame_data,  # message，message
            "uav_position": frame_data["uav_position"].copy(),
            "uav_orientation_quaternion": frame_data["uav_orientation_quaternion"].copy(),
            "uav_orientation_euler": frame_data["uav_orientation_euler"].copy()
        }
        
        merged_trajectory_data.append(frame_data)
    
    
    def _move_to_target_frame(self, u_target, t_target, i, num_steps, yaw_rate=None, jump_threshold=10.0):
        try:
            if i == 0:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"messageUAVmessage: {e}")
                current_orientation = uav_state['orientation']
                quat = airsim.Quaternionr(
                    w_val=current_orientation[0],
                    x_val=current_orientation[1],
                    y_val=current_orientation[2],
                    z_val=current_orientation[3]
                )
            else:
                uav_state = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        uav_state = self.get_uav_state()
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        else:
                            raise RuntimeError(f"messageUAVmessage: {e}")
                cur_pos = uav_state['position']
                current_orientation = uav_state['orientation']
                
                jump_distance = np.linalg.norm(np.array([u_target[0], u_target[1], u_target[2]]) - cur_pos)
                if jump_distance > jump_threshold:
                    error_msg = f"message（{jump_distance:.2f}m > {jump_threshold}m）message{i}message，message"
                    safe_log(f"✗ {error_msg}", scene_id=self.scene_id)
                    raise RuntimeError(error_msg)
                else:
                    if yaw_rate is not None:
                        from scipy.spatial.transform import Rotation as R
                        current_quat = [current_orientation[1], current_orientation[2], current_orientation[3], current_orientation[0]]  # [x, y, z, w]
                        rot = R.from_quat(current_quat)
                        euler = rot.as_euler('xyz', degrees=False)
                        current_yaw = euler[2]  # yaw message z message
                        dt = 1.0  # message
                        yaw_change = np.radians(yaw_rate * dt)  # message
                        new_yaw = current_yaw + yaw_change
                        new_euler = [euler[0], euler[1], new_yaw]  # [roll, pitch, yaw]
                        new_rot = R.from_euler('xyz', new_euler, degrees=False)
                        new_quat = new_rot.as_quat()  # [x, y, z, w]
                        quat = airsim.Quaternionr(
                            w_val=new_quat[3],
                            x_val=new_quat[0],
                            y_val=new_quat[1],
                            z_val=new_quat[2]
                        )
                    else:
                        dx = float(u_target[0]) - cur_pos[0]
                        dy = float(u_target[1]) - cur_pos[1]
                        yaw = np.arctan2(dy, dx)  # message
                        
                        quat = airsim.to_quaternion(0, 0, yaw)
            
            max_position_retries = 3
            ok, verify_pos, pos_error, err_xy, err_z = self._set_vehicle_pose_paused(
                float(u_target[0]), float(u_target[1]), float(u_target[2]),
                quat,
                retries=max_position_retries,
                tol_xy=0.5,
                tol_z=0.5
            )
            if not ok:
                error_msg = (
                    f"message：message({u_target[0]:.2f}, {u_target[1]:.2f}, {u_target[2]:.2f})，"
                    f"message({verify_pos[0]:.2f}, {verify_pos[1]:.2f}, {verify_pos[2]:.2f})，"
                    f"message{pos_error:.2f}m (XY:{err_xy:.2f}m, Z:{err_z:.2f}m)"
                )
                raise RuntimeError(error_msg)

            
            self.reset_collision_info()
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "streamclosederror" not in error_msg and "connection" not in error_msg:
                safe_log(f"⚠ message（message）: {e}", scene_id=self.scene_id)
        
        self.move_target_object(t_target)
        self._step_if_needed(1)
    
    def _process_frame(self, i, uav_traj, target_traj, trajectory_name, num_steps,
                      save_dataset, dataset_dir, merged_trajectory_data, pbar, target_trajectory_airsim=None):
        u_target = np.array([uav_traj[i][0], uav_traj[i][1], uav_traj[i][2]])
        t_target = np.array([target_traj[i][0], target_traj[i][1], target_traj[i][2]])
        
        jump_threshold = getattr(self, '_jump_threshold', 10.0)
        self._move_to_target_frame(u_target, t_target, i, num_steps, jump_threshold=jump_threshold)
        
        uav_state = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                uav_state = self.get_uav_state()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    safe_log(f"⚠ message：messageUAVmessage，message: {e}", scene_id=self.scene_id)
                    uav_state = {
                        'position': u_target.copy(),  # message
                        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # message
                        'has_collided': False
                    }
                    break
        
        cur1_pos = uav_state['position']
        
        pos2_now = None
        max_retries = 3
        for retry in range(max_retries):
            try:
                pos2_now = self.get_object_position()
                if pos2_now is not None:
                    break
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        if pos2_now is None:
            pos2_now = t_target.copy()  # message
        
        traj_num = trajectory_name.replace('trajectory_', '') if 'trajectory_' in trajectory_name else trajectory_name
        
        next_target_pos_airsim = None
        if i + 1 < len(target_traj):
            next_target_pos_airsim = np.array([
                target_traj[i + 1][0],
                target_traj[i + 1][1],
                target_traj[i + 1][2]
            ])
        
        if uav_state.get('has_collided', False):
            pbar.set_postfix_str("", refresh=True)
            tqdm.write("", file=sys.stderr)
            safe_log(f"⚠ message{traj_num}message，message{i}message", scene_id=self.scene_id)
            raise RuntimeError(f"message{traj_num}message{i}message")
        
        distance = np.linalg.norm(cur1_pos - pos2_now) if pos2_now is not None else 0
        
        try:
            if pos2_now is not None:
                pbar.set_postfix_str(
                    f"i={i}/{num_steps-1} "
                    f"D1=({cur1_pos[0]:.1f},{cur1_pos[1]:.1f},{-cur1_pos[2]:.1f}) "
                    f"T=({pos2_now[0]:.1f},{pos2_now[1]:.1f},{-pos2_now[2]:.1f}) "
                    f"dist={distance:.1f}m",
                    refresh=False  # message tqdm message mininterval message
                )
            else:
                pbar.set_postfix_str(
                    f"i={i}/{num_steps-1} "
                    f"D1=({cur1_pos[0]:.1f},{cur1_pos[1]:.1f},{-cur1_pos[2]:.1f}) "
                    f"T=N/A",
                    refresh=False
                )
        except Exception:
            pass
        
        if save_dataset:
            rgb_img, depth_img = self.get_camera_images()
            self.save_frame_data(i, rgb_img, depth_img, dataset_dir)
        
        if target_trajectory_airsim is not None and pos2_now is not None:
            target_trajectory_airsim.append({
                "x": float(pos2_now[0]),
                "y": float(pos2_now[1]),
                "z": float(-pos2_now[2])  # message：message NED(z<0) message
            })
        
        self._append_trajectory_data(i, uav_state, cur1_pos, pos2_now, 
                                     merged_trajectory_data, next_target_pos_airsim=next_target_pos_airsim)
    
    def _save_trajectory_files(self, dataset_dir, num_steps, selected_uav_name, 
                              merged_trajectory_data, save_dataset, target_trajectory_airsim=None,
                              planer_target_num_frames=None):
        if save_dataset:
            dataset_path = Path(dataset_dir)
            
            try:
                uav_traj_path = dataset_path / "uav_trajectory.json"
                temp_uav_path = dataset_path / "uav_trajectory.json.tmp"
                with open(temp_uav_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "num_frames": num_steps,
                        "target_asset_name": selected_uav_name,
                        "trajectory": merged_trajectory_data
                    }, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_uav_path.replace(uav_traj_path)
            except Exception as e:
                safe_log(f"⚠ message：messageUAVmessage: {e}", scene_id=self.scene_id)
                raise
            
            if target_trajectory_airsim is not None and len(target_trajectory_airsim) > 0:
                try:
                    target_traj_path = dataset_path / "target_trajectory.json"
                    temp_target_path = dataset_path / "target_trajectory.json.tmp"
                    target_num_frames = planer_target_num_frames if planer_target_num_frames is not None else len(target_trajectory_airsim)
                    
                    target_trajectory_save = []
                    for pos in target_trajectory_airsim:
                        if isinstance(pos, dict):
                            z_value = pos["z"]
                            if z_value < 0:
                                z_value = -z_value  # message z message，message（message）
                            target_trajectory_save.append({
                                "x": float(pos["x"]),
                                "y": float(pos["y"]),
                                "z": float(z_value)
                            })
                        else:
                            z_value = pos[2]
                            if z_value < 0:
                                z_value = -z_value  # message z message，message（message）
                            target_trajectory_save.append({
                                "x": float(pos[0]),
                                "y": float(pos[1]),
                                "z": float(z_value)
                            })
                    
                    with open(temp_target_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "num_frames": target_num_frames,
                            "target_trajectory_airsim": target_trajectory_save
                        }, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    temp_target_path.replace(target_traj_path)
                except Exception as e:
                    safe_log(f"⚠ message：message: {e}", scene_id=self.scene_id)
    
    def _check_final_distance(self, trajectory_name, uav_traj, target_traj):
        traj_num = trajectory_name.replace('trajectory_', '') if 'trajectory_' in trajectory_name else trajectory_name
        
        num_steps = min(len(uav_traj), len(target_traj))
        if num_steps == 0:
            return
        
        pos1_final_arr = np.array([
            uav_traj[num_steps - 1][0],
            uav_traj[num_steps - 1][1],
            uav_traj[num_steps - 1][2]
        ])
        
        try:
            if hasattr(self, '_last_uav_position') and self._last_uav_position is not None:
                pos1_final_arr = self._last_uav_position
            else:
                try:
                    uav_state_final = self.get_uav_state()
                    pos1_final_arr = uav_state_final['position']
                    self._last_uav_position = pos1_final_arr
                except Exception:
                    pass
        except Exception:
            pass
        
        pos2_final_arr = np.array([
            target_traj[num_steps - 1][0],
            target_traj[num_steps - 1][1],
            target_traj[num_steps - 1][2]
        ])
        
        final_distance = np.linalg.norm(pos1_final_arr - pos2_final_arr)
        if final_distance >= 5.0:
            if hasattr(self, '_current_pbar') and self._current_pbar is not None:
                self._current_pbar.set_postfix_str("", refresh=True)
            tqdm.write("", file=sys.stderr)
            safe_log(f"⚠ message{traj_num}message，message5m（{final_distance:.2f}m）", scene_id=self.scene_id)
            raise RuntimeError(f"message{traj_num}message，message5m（{final_distance:.2f}m）")
    
    def _cleanup_after_execution(self, skip_hover):
        if not skip_hover:
            try:
                print("\nmessage... (message Ctrl+C message)")
                print("  message:")
                while True:
                    uav_state = self.get_uav_state()
                    pos1_arr = uav_state['position']
                    pos2_arr = self.get_object_position()
                    
                    if pos2_arr is not None:
                        dx = pos2_arr[0] - pos1_arr[0]
                        dy = pos2_arr[1] - pos1_arr[1]
                        yaw = np.arctan2(dy, dx)  # message
                        current_distance = np.linalg.norm(pos1_arr - pos2_arr)
                        print(f"    {current_distance:.2f} m", end='\r')
                    else:
                        orientation = uav_state['orientation']
                        yaw = 0.0  # message
                    
                    quat = airsim.to_quaternion(0, 0, yaw)
                    
                    self.client.simSetVehiclePose(
                        airsim.Pose(airsim.Vector3r(pos1_arr[0], pos1_arr[1], pos1_arr[2]), quat),
                        ignore_collision=True,
                        vehicle_name=self.uav_vehicle_name
                    )
            except KeyboardInterrupt:
                print("\n\nmessage，message...")
            
            print("\n✓ message，AirSim message")
        
        if self.client is not None:
            try:
                self.client.simDestroyObject(self.target_object_name)
            except Exception as e:
                print(f"⚠ message: {e}")
                try:
                    existing_objects = self.client.simListSceneObjects(self.target_object_name + ".*")
                    if existing_objects:
                        for obj_name in existing_objects:
                            try:
                                self.client.simDestroyObject(obj_name)
                                print(f"  message: {obj_name}")
                            except:
                                pass
                except:
                    pass
    
    def execute_trajectory(self, trajectory_file, dataset_base_dir="/mnt/Data20T/ysq/OurVLN/Dataset", save_dataset=True, skip_hover=False, trajectory_index=None, total_trajectories=None, max_retries=5, jump_threshold=10.0):
        if not os.path.exists(trajectory_file):
            print(f"message：message：{trajectory_file}")
            return
        
        uav_traj, target_traj = self.load_trajectory(trajectory_file)
        
        if save_dataset:
            trajectory_name = Path(trajectory_file).stem
            if trajectory_name.endswith('_uav'):
                trajectory_name = trajectory_name[:-4]
            elif trajectory_name.endswith('_target'):
                trajectory_name = trajectory_name[:-7]
            
            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
            rgb_dir = dataset_path / "rgb"
            uav_json_file = dataset_path / "uav_trajectory.json"
            
            num_steps = min(len(uav_traj), len(target_traj))
            if rgb_dir.exists() and uav_json_file.exists():
                existing_frames = []
                for frame_file in rgb_dir.glob("frame_*.png"):
                    try:
                        frame_num = int(frame_file.stem.split('_')[1])
                        existing_frames.append(frame_num)
                    except:
                        continue
                
                if len(existing_frames) >= num_steps:
                    expected_frames = set(range(num_steps))
                    saved_frames = set(existing_frames)
                    if expected_frames.issubset(saved_frames):
                        safe_log(f"⏭ [{self.scene_id}] message {trajectory_name} message，message", scene_id=self.scene_id)
                        return
                    else:
                        safe_log(f"🔄 [{self.scene_id}] message {trajectory_name} message（message{len(existing_frames)}/{num_steps}message），message", scene_id=self.scene_id)
                        try:
                            depth_dir = dataset_path / "depth"
                            for frame_file in rgb_dir.glob("frame_*.png"):
                                try:
                                    frame_file.unlink()
                                except:
                                    pass
                            if depth_dir.exists():
                                for frame_file in depth_dir.glob("frame_*.png"):
                                    try:
                                        frame_file.unlink()
                                    except:
                                        pass
                            for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'instruction.json']:
                                json_path = dataset_path / json_file
                                if json_path.exists():
                                    try:
                                        json_path.unlink()
                                    except:
                                        pass
                        except Exception as e:
                            safe_log(f"⚠ [{self.scene_id}] message: {e}", scene_id=self.scene_id)
                elif len(existing_frames) > 0:
                    safe_log(f"🔄 [{self.scene_id}] message {trajectory_name} message（message{len(existing_frames)}/{num_steps}message），message", scene_id=self.scene_id)
                    try:
                        depth_dir = dataset_path / "depth"
                        for frame_file in rgb_dir.glob("frame_*.png"):
                            try:
                                frame_file.unlink()
                            except:
                                pass
                        if depth_dir.exists():
                            for frame_file in depth_dir.glob("frame_*.png"):
                                try:
                                    frame_file.unlink()
                                except:
                                    pass
                        for json_file in ['uav_trajectory.json', 'target_trajectory.json', 'instruction.json']:
                            json_path = dataset_path / json_file
                            if json_path.exists():
                                try:
                                    json_path.unlink()
                                except:
                                    pass
                    except Exception as e:
                        safe_log(f"⚠ [{self.scene_id}] message: {e}", scene_id=self.scene_id)
        
        self._abnormal_jumps = []
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                return self._execute_trajectory_internal(trajectory_file, dataset_base_dir, save_dataset, skip_hover, trajectory_index, total_trajectories, uav_traj, target_traj, jump_threshold=jump_threshold)
            except RuntimeError as e:
                error_msg = str(e)
                if "message" in error_msg or "collision" in error_msg.lower() or "message5m" in error_msg or "message" in error_msg:
                    retry_count += 1
                    if retry_count <= max_retries:
                        safe_log(f"🔄 message{retry_count}message（message：{error_msg}）", scene_id=self.scene_id)
                        try:
                            self._cleanup_after_execution(skip_hover=True)
                        except:
                            pass
                        try:
                            trajectory_name = Path(trajectory_file).stem
                            if trajectory_name.endswith('_uav'):
                                trajectory_name = trajectory_name[:-4]
                            elif trajectory_name.endswith('_target'):
                                trajectory_name = trajectory_name[:-7]
                            dataset_path = Path(dataset_base_dir) / self.scene_id / trajectory_name
                            for json_file in ['uav_trajectory.json', 'target_trajectory.json']:
                                json_path = dataset_path / json_file
                                if json_path.exists():
                                    json_path.unlink()
                        except:
                            pass
                        import time
                        time.sleep(0.5)
                        continue
                    else:
                        safe_log(f"✗ message：message（{max_retries}message），message", scene_id=self.scene_id)
                        return
                else:
                    raise
            except Exception as e:
                raise
    
    def _execute_trajectory_internal(self, trajectory_file, dataset_base_dir, save_dataset, skip_hover, trajectory_index, total_trajectories, uav_traj, target_traj, jump_threshold=10.0):
        
        self._jump_threshold = jump_threshold
        
        self._abnormal_jumps = []
        
        selected_uav_name = self._prepare_target_object()
        
        trajectory_name = Path(trajectory_file).stem
        if trajectory_name.endswith('_uav'):
            trajectory_name = trajectory_name[:-4]  # message '_uav'
        elif trajectory_name.endswith('_target'):
            trajectory_name = trajectory_name[:-7]  # message '_target'
        
        planer_target_num_frames = None
        planer_target_positions_airsim = None  # message Planer dataset message（AirSim message）
        try:
            trajectory_path = Path(trajectory_file)
            if trajectory_path.name.endswith('_uav.json') or trajectory_path.name.endswith('_target.json'):
                base_name = trajectory_path.name.replace('_uav.json', '').replace('_target.json', '')
                planer_target_file = trajectory_path.parent / f"{base_name}_target.json"
            else:
                planer_target_file = trajectory_path
            
            if planer_target_file.exists():
                with open(planer_target_file, 'r', encoding='utf-8') as f:
                    planer_target_data = json.load(f)
                if 'target_trajectory' in planer_target_data and isinstance(planer_target_data['target_trajectory'], list):
                    planer_target_num_frames = len(planer_target_data['target_trajectory'])
                    planer_target_positions_airsim = []
                    for pos in planer_target_data['target_trajectory']:
                        if isinstance(pos, list) and len(pos) >= 3:
                            airsim_pos = {
                                "x": float(pos[0]),
                                "y": float(-pos[1]),  # y message
                                "z": float(-pos[2])   # z message
                            }
                            planer_target_positions_airsim.append(airsim_pos)
        except Exception as e:
            safe_log(f"⚠ message：message Planer target message: {e}，message", scene_id=self.scene_id)
        
        dataset_dir = self._prepare_dataset_directory(trajectory_name, dataset_base_dir, save_dataset)
        
        
        self._initialize_simulation(uav_traj, target_traj)
        
        self._reset_collision_state()
        
        num_steps = min(len(uav_traj), len(target_traj))
        merged_trajectory_data = []
        
        target_trajectory_airsim = []
        
        self._prev_frame_data = None
        
        try:
            if not self.client.isApiControlEnabled(vehicle_name=self.uav_vehicle_name):
                max_retries = 5
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        self.client.enableApiControl(True, vehicle_name=self.uav_vehicle_name)
                        self.client.armDisarm(True, vehicle_name=self.uav_vehicle_name)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            error_msg = str(e)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                time.sleep(retry_delay)
                                continue
                        raise
        except:
            pass
        
        self._ensure_uav_flying_state()
        
        progress_position = getattr(self, '_progress_position', None)
        if trajectory_index is not None and total_trajectories is not None:
            desc = f"[{self.scene_id}] message{trajectory_index}/{total_trajectories}"
        else:
            desc = f"[{self.scene_id}]"
        
        if progress_position is not None:
            pbar = tqdm(range(0, num_steps), 
                       desc=desc, 
                       unit="message", 
                       position=progress_position, 
                       leave=True, 
                       file=sys.stderr,
                       dynamic_ncols=True,
                       mininterval=0.1,
                       maxinterval=1.0)
            self._current_pbar = pbar
        else:
            pbar = tqdm(range(0, num_steps), 
                       desc=desc if trajectory_index is not None else "message", 
                       unit="message",
                       file=sys.stderr,
                       dynamic_ncols=True,
                       mininterval=0.1)
            self._current_pbar = pbar
        
        for i in pbar:
            try:
                self._process_frame(i, uav_traj, target_traj, trajectory_name, num_steps,
                                  save_dataset, dataset_dir, merged_trajectory_data, pbar, target_trajectory_airsim)
                
            except RuntimeError as e:
                error_msg = str(e)
                if "message" in error_msg or "collision" in error_msg.lower() or "message" in error_msg or "message" in error_msg:
                    safe_log(f"✗ message：Step {i} message: {e}", scene_id=self.scene_id)
                    try:
                        pbar.close()
                    except:
                        pass
                    raise
                else:
                    safe_log(f"✗ message：Step {i} message: {e}", scene_id=self.scene_id)
                    continue
            except Exception as e:
                safe_log(f"✗ message：Step {i} message: {e}", scene_id=self.scene_id)
                import traceback
                import io
                traceback_str = io.StringIO()
                traceback.print_exc(file=traceback_str)
                safe_log(traceback_str.getvalue(), scene_id=self.scene_id)
                continue
        
        try:
            if pbar is not None:
                pbar.refresh()
                pbar.close()
        except:
            pass
        finally:
            if hasattr(self, '_current_pbar'):
                self._current_pbar = None
        
        try:
            if num_steps > 0:
                self._last_uav_position = np.array([
                    uav_traj[num_steps - 1][0],
                    uav_traj[num_steps - 1][1],
                    uav_traj[num_steps - 1][2]
                ])
                try:
                    uav_state = self.get_uav_state()
                    self._last_uav_position = uav_state['position']
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            target_positions_to_save = planer_target_positions_airsim if planer_target_positions_airsim is not None else target_trajectory_airsim
            self._save_trajectory_files(dataset_dir, num_steps, selected_uav_name, 
                                       merged_trajectory_data, save_dataset, target_positions_to_save,
                                       planer_target_num_frames=planer_target_num_frames)
        except Exception as e:
            safe_log(f"⚠ message：message: {e}", scene_id=self.scene_id)
            import traceback
            traceback.print_exc()
        
        self._check_final_distance(trajectory_name, uav_traj, target_traj)
        
        try:
            self._cleanup_after_execution(skip_hover)
        except Exception as e:
            safe_log(f"⚠ message：message: {e}", scene_id=self.scene_id)

