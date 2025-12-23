from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Callable
import re
from loguru import logger

_FUNIT = {
    'hz': 1,
    'khz': 1000,
    'mhz': 1e6,
    'ghz': 1e9,
}
_NPORTS = {
    '.s1p': 1,
    '.s2p': 2,
    '.s3p': 3,
    '.s4p': 4,
    '.snp': None
}


############################################################
#                         FUNCTIONS                        #
############################################################


def _ma_ri(mag: float, angle: float) -> complex:
    """Converts magnitude and angle to complex number.

    Args:
        mag (float): The magnitude of the complex number.
        angle (float): The angle of the complex number.

    Returns:
        complex: The complex number.
    """    
    return mag * np.exp(1j * np.radians(angle))

def _db_ri(db: float, angle: float) -> complex:
    """Converts dB and angle to complex number.

    Args:
        db (float): The magnitude in dB.
        angle (float): The angle in degrees.

    Returns:
        complex: The complex number.
    """    
    return 10**(db/20) * np.exp(1j * np.radians(angle))

def _ri_ri(real: float, imag: float) -> complex:
    """Converts real and imaginary parts to complex number.

    Args:
        real (float): The real part of the complex number.
        imag (float): The imaginary part of the complex number.

    Returns:
        complex: The complex number.
    """    
    return real + 1j * imag

_DATAMAP = {
    'ma': _ma_ri,
    'db': _db_ri,
    'ri': _ri_ri
}


############################################################
#                        MAIN CLASS                       #
############################################################


class TouchstoneData:
    """A touchstone file importer for S-parameter data."

    The TouchstoneData class can be used to quickly import Touchstone files into your model to plot the data in your graphs.
    
    Args:
        filename (str): The filename of the touchstone file.
        n_ports (int, optional): The number of ports. Defaults to None.
        ignore_extension (bool, optional): Ignore the file extension. Defaults to False.
        interp_type (str, optional): The interpolation type. Defaults to 'cubic'.

    Example:
        ```python
        touchstone = ToustoneData('touchstone.s2p')
        touchstone.renormalize(75)
        ```

    """
    def __init__(self, filename: str,
                 n_ports: int = None,
                 ignore_extension: bool = False,
                 interp_type: str = 'cubic'):
        super().__init__()
        path = Path(filename)
        if not path.exists():
            raise ValueError(f'The provided filename {path} does not exist.')

        if path.suffix not in ('.s1p','.s2p','.s3p','.s4p','.snp','.ts') and not ignore_extension:
            raise ValueError(f'Can only open touchstone files, files with {path.suffix} are not a valid touchstone extension.')
        
        with open(filename,'r') as f:
            self._lines = f.read().split('\n')
        
        self.Sdata = None

        self.freq_unit: float = 1e9
        self.param_type: str = 's'
        self.data_type: str = 'ma'
        self._converter: Callable = _DATAMAP['ma']
        self.refimp: float = 50
        self.n_ports: int = None
        self.interp_type: str = interp_type
        self.Sdata: np.ndarray = None
        self.Fdata: np.ndarray = None

        if self.param_type != 's':
            raise ValueError('Only S-parameter touchstone files are supported as of this moment.')

        self._parse_touchstone()
    
    @property 
    def f(self) -> np.ndarray:
        return self.Fdata
    
    def summarize_data(self):
        print(f'Frequency unit: {self.freq_unit}')
        print(f'Parameter type: {self.param_type}')
        print(f'Data type: {self.data_type}')
        print(f'Converter: {self._converter}')
        print(f'Reference impedance: {self.refimp}')
        print(f'Number of ports: {self.n_ports}')

    def _parse_touchstone(self) -> None:
        s_data = []
        f_data = []
        s_collector = []
        freq = None
        for line in self._lines:
            if not line:
                continue
            
            stripped = line.strip().lower()
            # Comment line
            if stripped[0]=='!':
                continue
            
            # Options line
            if stripped[0]=='#':
                funit = re.findall(r'(hz|khz|mhz|ghz)', stripped)
                param_type = re.findall(r'([syzgh])', stripped)
                data_type = re.findall(r'(ma|db|ri)', stripped)
                refimp = re.findall(r'r\s+(\d+)', stripped)

                if funit:
                    self.freq_unit = _FUNIT[funit[0]]
                if param_type:
                    self.param_type = param_type[0]
                if data_type:
                    self.data_type = data_type[0]
                    self._converter = _DATAMAP[data_type[0]]
                if refimp:
                    self.refimp = float(refimp[0])
            
                continue
            
            # Version 2.0 Keyword line:
            if stripped[0]=='[':
                logger.warning('Version 2.0 touchstone files are not supported yet and keywords will be ignored')
                continue
            # Data line
            # remove the exclamation mark plus all symbols behind it if it occurs
            line = line.split('!')[0]

            nums = [float(x) for x in line.split(' ') if x]

            if len(nums)%2 == 1:
                if len(s_collector) > 0:
                    s_data.append(s_collector)
                    f_data.append(freq)
                s_collector = nums[1:]
                freq = nums[0]
            else:
                s_collector.extend(nums)
            
        s_data.append(s_collector)
        f_data.append(freq)

        ssample = s_data[0]
        self.n_ports = int(np.sqrt(len(ssample) // 2))
        self.n_nodes = self.n_ports

        self.n_frequencies = len(f_data)
        self.Fdata = np.array(f_data) * self.freq_unit
        self.Sdata = np.zeros((self.n_frequencies, self.n_ports, self.n_ports), dtype=complex)

        for i, s in enumerate(s_data):
            sdata = [self._converter(s1,s2) for s1,s2 in zip(s[::2], s[1::2])]
            Smat = np.array(sdata).reshape(self.n_ports, self.n_ports)
            if self.n_ports == 2:
                Smat = Smat.T
            self.Sdata[i,:,:] = Smat

    def renormalize(self, new_impedance: float) -> TouchstoneData:
        """ Renormalize the touchstone file to a new impedance.

        Args:
            new_impedance (float): The new impedance to normalize the touchstone file to.

        Returns:
            FileBasedNPort: The renormalized touchstone file subcircuit.
        """        
        Zold = self.refimp
        Znew = new_impedance
        A = np.eye(self.n_ports, self.n_ports) * np.sqrt(Znew/Zold)*(1/(Znew+Zold))
        R = np.eye(self.n_ports, self.n_ports) * ((Znew-Zold)/(Znew+Zold))
        iA = np.linalg.pinv(A)
        E = np.eye(self.n_ports, self.n_ports)
        for iif in range(self.n_frequencies):
            S = self.Sdata[iif,:,:]
            self.Sdata[iif,:,:] = iA @ (S - R) @ np.linalg.pinv(E - R @ S) @ A
        self.refimp = new_impedance
        return self
        
    def S(self, i: int, j: int) -> np.ndarray:
        """Return the S-parameter array corresponding to the given port number

        Args:
            i (int): The output port
            j (int): The input Port

        Returns:
            np.ndarray: The S-parameter array
        """
        return self.Sdata[:, i-1, j-1].squeeze()