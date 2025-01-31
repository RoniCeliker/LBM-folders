import sys
import re
import os

import numpy as np


class FFA:
    """
    name = "Default"
    type = "N   "
    data = np.empty((0,0)))
    sub  = []
    ndim = 0
    nsiz = 0
    nsub = 0
    """
    type_dict = {
        "N": None,
        "I": "int32",
        "J": "int64",
        "R": "float32",
        "D": "float64",
        "C": "complex64",
        "Z": "complex128",
        "A": "|1S",
        "S": "|16S",
        "L": "|72S",
    }

    def __init__(self, m_name="Default", m_data=None, m_type=None):
        self.name = m_name
        if m_data is None:
            self.data = np.empty((0, 0))
        else:
            self.data = m_data

        if m_type is None:
            if m_data is None:
                self.type = "N   "
            elif self.data.dtype.type is np.int32:
                self.type = "I   "
            elif self.data.dtype.type is np.int64:
                self.type = "J   "
            elif self.data.dtype.type is np.float32:
                self.type = "R   "
            elif self.data.dtype.type is np.float64:
                self.type = "D   "
            elif self.data.dtype.type is np.str_:
                self.type = "L   "
        else:
            self.type = m_type
        self.sub  = []
        self.comments = []

    def __str__(self):
        return self.__list()

    def __iter__(self):
        return self.sub.__iter__()

    def __setattr__(self, key, value):
        if key == "data":
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            rank = len(value.shape)
            if rank == 0 or rank == 1:
                value = np.reshape(value, (-1, 1))
            elif rank == 2:
                pass
            else:
                raise TypeError('In dataset "%s", array data with '
                                'rank=%i is not allowed' % (self.name, rank))
        elif key == "name":
            if sys.version_info.major == 2:
                value = str(value)
            if isinstance(value, str):
                if len(value) > 16:
                    value = "%-16s" % value
                    value = value[0:16]
            else:
                raise TypeError('name must be of "str" type')
        elif key == "type":
            if sys.version_info.major == 2:
                value = str(value)
            if isinstance(value, str):
                value = "%-4s" % value
                value = value[0:4]
                if value[0] not in self.type_dict.keys():
                    types = "".join([key for key in self.type_dict.keys()])
                    raise TypeError('type[0] must be in "%s"' % types)
                # the type value governs the data type
                self.data = self.data.astype(self.type_dict[value[0]])
                if value[0] in "ASL":
                    self.data = self.data.astype(str)
            else:
                raise TypeError('type must be of "str" type')
        elif key == "sub":
            if isinstance(value, FFA):
                value = [value]
            if isinstance(value, list):
                if not all(isinstance(sub, FFA) for sub in value):
                    raise TypeError('sub list must only contain "FFA" type')
            else:
                raise TypeError('sub must be of "list" type')
        self.__dict__[key] = value

    def __repr__(self):
        return 'FFA("%s")' % self.name.strip()

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.ndim != other.ndim:
            return False
        if self.nsiz != other.nsiz:
            return False
        if self.nsub != other.nsub:
            return False
        if not np.alltrue(self.data == other.data):
            return False
        for i in range(0, self.nsub):
            if self.sub[i] != other.sub[i]:
                return False
        return True

    @property
    def ndim(self):
        return np.int64(self.data.shape[1])

    @property
    def nsiz(self):
        return np.int64(self.data.shape[0])

    @property
    def nsub(self):
        return np.int64(len(self.sub))

    def __list(self, indent=3, level=0):
        s = ""
        s = "%s%-4s " % (s, self.type)
        s = "%s%9i x %-6i " % (s, self.nsiz, self.ndim)
        if self.nsub == 0:
            s = "%s%4s " % (s, "")
        else:
            s = "%s%3i/ " % (s, self.nsub)
        s = "%s%s" % (s, indent*level*" ")
        s = "%s%-16s " % (s, self.name)
        if self.ndim*self.nsiz > 0:
            if self.type[0] in "ASL":
                s = '%s = "%s"' % (s, self.data[0, 0].strip())
            elif self.type[0] in "IJ":
                if self.ndim*self.nsiz >= 3:
                    tmp = self.data.T.flatten()[0:3]
                    s = "%s = % 12i    % 12i    % 12i" % (s, tmp[0], tmp[1], tmp[2])
                else:
                    s = "%s = % 12i" % (s, self.data[0, 0])
            elif self.type[0] in "RD":
                if self.ndim*self.nsiz >= 3:
                    tmp = self.data.T.flatten()[0:3]
                    s = "%s = % 12g    % 12g    % 12g" % (s, tmp[0], tmp[1], tmp[2])
                else:
                    s = "%s = % 12g" % (s, self.data[0, 0])
        return s

    def __read_recursive(self, fi, version, skip=""):
        nnn = self.__read_descriptor(fi, version)
        self.__read_data(fi, version, nnn, skip)
        nsub = nnn[2]
        for i in range(0, nsub):
            self.sub.append(FFA())
            self.sub[-1].__read_recursive(fi, version, skip)

    def __read_descriptor(self, fi, version):
        if version == 2:
            paci = ">q"  # Big endian 8 byte integer
            self.name = np.fromfile(fi, dtype="|16S", count=1)[0].decode()
            self.type = np.fromfile(fi, dtype="|4S",  count=1)[0].decode()
            nnn       = np.fromfile(fi, dtype=paci,   count=3)
        elif version == 1:
            paci = ">i"  # Big endian 4 byte integer
            np.fromfile(fi, dtype=paci, count=1)
            self.name = np.fromfile(fi, dtype="|16S", count=1)[0].decode()
            self.type = np.fromfile(fi, dtype="|4S",  count=1)[0].decode()
            nnn       = np.fromfile(fi, dtype=paci,   count=3)
            np.fromfile(fi, dtype=paci, count=1)
        elif version == 0:
            line = self.__read_ffa_ascii_line(fi)
            line.reverse()
            self.name = line.pop()
            self.type = line.pop()
            nnn = [int(line.pop()), int(line.pop()), int(line.pop())]
            if len(line) != 0:
                raise IOError('Unexpected trailing field "%s" in FFA header' % str(line))

        else:
            raise TypeError("Version %i is not applicable." % version)
        return nnn

    def __read_ffa_ascii_line(self, fi):
        line = []
        while len(line) == 0:
            linei = fi.readline()
            if linei == '':
                raise IOError('Unexpected end of file')
            if re.match(r'\s*\*.*', linei):
                self.comments.append(linei)
                continue
            line = re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s]+', linei)
        return line

    @staticmethod
    def __ffa_str2str(s):
        if re.match(r'\s*\'[^\']*\'\s*$', s):
            return re.sub(r'\s*\'([^\']*)\'\s*$', r'\g<1>', s)
        elif re.match(r'\s*"[^"]*"\s*$', s):
            return re.sub(r'\s*"([^"]*)"\s*$', r'\g<1>', s)
        else:
            raise ValueError('Invalid string "%s"' % s)

    def __read_data(self, fi, version, nnn, skip=""):
        ndim = nnn[0]
        nsiz = nnn[1]
        count = ndim*nsiz
        if count == 0:
            self.data = np.empty((0, 0))
            return
        if self.type[0] == "N":
            self.data = np.empty((0, 0))
        elif self.type[0] in self.type_dict.keys():
            if version == 2:
                beg = np.fromfile(fi, dtype=">q", count=1)[0]
                if self.type[1] in skip:
                    fi.seek(fi.tell() + beg)
                    self.data = np.empty((0, 0))
                else:
                    self.data = np.fromfile(
                        fi,
                        dtype=self.type_dict[self.type[0]],
                        count=count,)
            elif version == 1:
                beg = np.fromfile(fi, dtype=">i", count=1)[0]
                if self.type[1] in skip:
                    fi.seek(fi.tell() + beg)
                    self.data = np.empty((0, 0))
                else:
                    self.data = np.fromfile(
                        fi,
                        dtype=self.type_dict[self.type[0]],
                        count=count,)
                np.fromfile(fi, dtype=">i", count=1)
            elif version == 0:
                data = np.empty(nsiz*ndim,
                                dtype=self.type_dict[self.type[0]])
                index = 0
                while index < nsiz*ndim:
                    line = fi.readline()
                    if self.type[0] in "ASL":
                        line = np.array(re.findall('\'[^\']*\'', line))
                    else:
                        line = np.array(re.findall(r'"[^"]*"|\'[^\']*\'|[^,\s]+', line))
                    for i in range(0, line.size):
                        if self.type[0] in "ASL":
                            data[index] = self.__ffa_str2str(line[i])
                        else:
                            data[index] = line[i]
                        index += 1
                self.data = data
                self.data = self.data.reshape((ndim, nsiz)).T
            else:
                raise TypeError("Version %i is not applicable." % version)
        else:
            raise IOError('Unrecognised descriptor "%s"' % self.type[0])
        # self.data = self.data.reshape((nsiz, ndim))
        self.data = self.data.reshape((ndim, nsiz)).T
        # byte swap if necessary
        if version != 0 and np.little_endian:
            self.data.byteswap(True)
        # decode string
        if self.type[0] in "ASL":
            if self.data.dtype.type is np.bytes_:
                self.data = self.data.astype(str)

    def __write_recursive(self, fi, version):
        self.__write_descriptor(fi, version)
        self.__write_data(fi, version)
        for i in range(0, self.nsub):
            self.sub[i].__write_recursive(fi, version)

    def __write_descriptor(self, fi, version):
        if version == 2:
            fi.write(("%-16s" % self.name).encode())
            fi.write(("%-4s" % self.type).encode())
            if np.little_endian:
                np.int64(self.ndim).byteswap().tofile(fi, sep="")
                np.int64(self.nsiz).byteswap().tofile(fi, sep="")
                np.int64(self.nsub).byteswap().tofile(fi, sep="")
            else:
                np.int64(self.ndim).tofile(fi, sep="")
                np.int64(self.nsiz).tofile(fi, sep="")
                np.int64(self.nsub).tofile(fi, sep="")

        elif version == 1:
            if np.little_endian:
                np.int32(32).byteswap().tofile(fi, sep="")
            else:
                np.int32(32).tofile(fi, sep="")
            fi.write(("%-16s" % self.name).encode())
            fi.write(("%-4s" % self.type).encode())
            if np.little_endian:
                np.int32(self.ndim).byteswap().tofile(fi, sep="")
                np.int32(self.nsiz).byteswap().tofile(fi, sep="")
                np.int32(self.nsub).byteswap().tofile(fi, sep="")
            else:
                np.int32(self.ndim).tofile(fi, sep="")
                np.int32(self.nsiz).tofile(fi, sep="")
                np.int32(self.nsub).tofile(fi, sep="")
            if np.little_endian:
                np.int32(32).byteswap().tofile(fi, sep="")
            else:
                np.int32(32).tofile(fi, sep="")
        elif version == 0:
            for line in self.comments:
                fi.write(line)
            fi.write("%s,%-4s,%i,%i,%i\n" %
                     (self.name, self.type, self.ndim, self.nsiz, self.nsub))

        else:
            raise TypeError("Version %i is not applicable." % version)

    def __write_data(self, fi, version):
        if self.type[0] == "N" or self.ndim*self.nsiz == 0:
            return
        if self.type[0] in "ASL":
            # Make sure we turn string into bytes when writing
            if self.data.dtype.type is np.str_:
                self.data = self.data.astype(bytes)
        size = self.ndim*self.nsiz*self.data.itemsize
        if version == 2:
            if np.little_endian:
                np.int64(size).byteswap().tofile(fi, sep="")
                self.data.T.byteswap().tofile(fi, sep="")
            else:
                np.int64(size).tofile(fi, sep="")
                self.data.T.tofile(fi, sep="")
        elif version == 1:
            if np.little_endian:
                np.int32(size).byteswap().tofile(fi, sep="")
                self.data.T.byteswap().tofile(fi, sep="")
                np.int32(size).byteswap().tofile(fi, sep="")
            else:
                np.int32(size).tofile(fi, sep="")
                self.data.T.tofile(fi, sep="")
                np.int32(size).tofile(fi, sep="")
        elif version == 0:
            for j in range(0, self.ndim):
                if self.type[0] in "ASL":
                    tmp = self.data[:, j].astype(str)
                    tmp = " ".join(np.char.mod("'%s'", tmp))
                else:
                    tmp = self.data[:, j]
                    tmp = " ".join(np.char.mod("%s", tmp))
                fi.write(tmp)
                fi.write('\n')
        else:
            raise TypeError("Version %i is not applicable." % version)

        if self.type[0] in "ASL":
            # Revert strings now
            if self.data.dtype.type is np.bytes_:
                self.data = self.data.astype(str)

    @staticmethod
    def __read(filename, skip="", fmt="binary"):
        ds = FFA()
        if not isinstance(filename, str):
            raise TypeError("Filename must be of type str")
        if not os.path.isfile(filename):
            raise IOError("File does not exist")

        if fmt == "binary":
            with open(filename, "rb") as fi:
                test = fi.read(32)
                if test == 'FFA-format-v2                   '.encode():
                    version = 2
                else:
                    version = 1
                    fi.seek(0)
                ds.__read_recursive(fi, version, skip)
        elif fmt == "ascii":
            with open(filename, "r") as fi:
                version = 0
                ds.__read_recursive(fi, version, skip)
        return ds

    def __write(self, filename, version=2):
        if version == 2:
            with open(filename, "wb") as fi:
                fi.write(("%-32s" % np.array("FFA-format-v2").astype(str)).encode())
                self.__write_recursive(fi, version)
        elif version == 1:
            with open(filename, "wb") as fi:
                self.__write_recursive(fi, version)
        elif version == 0:
            with open(filename, "w") as fi:
                self.__write_recursive(fi, version)
        else:
            raise TypeError("Version %i is not applicable." % version)

    def list(self, level_break=-1, indent=3, level=0):
        print(self.__list(indent, level))
        for i in range(0, self.nsub):
            if level_break == -1 or level < level_break:
                self.sub[i].list(level_break, indent, level+1)

    def append(self, subset):
        if subset is None:
            return
        if isinstance(subset, FFA):
            self.sub.append(subset)
        else:
            raise TypeError('Argument to "append" has wrong type')

    def insert(self, subset, index=0):
        if subset is None:
            return
        if isinstance(subset, FFA):
            self.sub.insert(index, subset)
        else:
            raise TypeError('Argument to "insert" has wrong type')

    def delete(self, subset):
        if subset is None:
            return
        if isinstance(subset, FFA):
            if subset in self.sub:
                self.sub.pop(self.sub.index(subset))
        elif isinstance(subset, int):
            self.sub.pop(subset)
        elif isinstance(subset, list):
            for s in subset:
                self.delete(s)
        else:
            raise TypeError('Argument to "delete" has wrong type')

    def get(self, name):
        for sub in self.sub:
            if sub.name.strip() == name:
                return sub

    def getl(self, name):
        index = np.full(self.nsub, False)
        for i in range(0, self.nsub):
            if self.sub[i].name.strip() == name:
                index[i] = True
        return [self.sub[i] for i in np.where(index)[0]]

    def gets(self, name):
        sub = []
        for i in range(0, self.nsub):
            if self.sub[i].name.strip() == name:
                sub.append(self.sub[i])
            else:
                sub += self.sub[i].gets(name)
        return sub

    @staticmethod
    def read(filename, skip=""):
        try:
            return FFA.__read(filename, skip=skip, fmt="binary")
        except:
            return FFA.__read(filename, skip=skip, fmt="ascii")

    def write(self, filename, version=2):
        return self.__write(filename, version)


def read(filename, skip=""):
    return FFA.read(filename, skip)


def load(filename, skip=""):
    return FFA.read(filename, skip)
