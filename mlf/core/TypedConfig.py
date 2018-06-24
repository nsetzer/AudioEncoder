import configparser
import hashlib
from collections import OrderedDict

class ConfigType(object):
    """
    This object is an enum containing different types of
    values that may be contained in configuration.
    """
    Unknown = 0x000
    Bool = 0x001
    Int = 0x002
    Float = 0x004
    String = 0x008
    Csv = 0x100  # when type cannot be inferred
    CsvBool = 0x101
    CsvInt = 0x102
    CsvFloat = 0x104
    CsvString = 0x108

    @staticmethod
    def toValue(type, value):
        """
        :param type: The type of the value
        :param value: The value as a string to convert to its python value
        :return: The pythonated value
        """
        if type & ConfigType.Bool:
            return bool(value)
        elif type & ConfigType.Int:
            return int(value)
        elif type & ConfigType.Float:
            return float(value)
        else:
            return str(value)

class TypedConfig(object):
    """
    Provides the ability to load and save experiment configurations
    Configuration variables can be manually grouped together in
    sections.

    Assumes lists/tuples are homogeneous types
    """

    def __init__(self):
        super(TypedConfig, self).__init__()
        # map to help group certain attributes together
        self._sections = {}
        # ignore these attributes when generating uid
        self._blacklist = set()

    def addGroup(self, groupName, members):
        """
        create a new section in the config containing the given members
        :param groupName: The name of the section to add
        :param members: The members to add
        """
        self._sections[groupName] = set(members)

    def updateBlacklist(self, attr_iter):
        """
        Add a single attribute or a list of attributes to the blacklist.

        The blacklist is used to mark attributes that should not be
        included when computing the unique identifier for this config

        :param attr_iter: A list of strings/lists-containing-strings to blacklist
        """
        if isinstance(attr_iter, str):
            self._blacklist.add(attr_iter)
        else:
            for attr in attr_iter:
                self._blacklist.add(attr)

    def sectionName(self):
        """
        :return: "default"
        """
        return "default"

    def getUngroupedAttributes(self):
        """
        :return: The set of ungrouped attributes on this object.
        """
        attrs = self._get_attrs()
        for members in self._sections.values():
            attrs -= members
        return attrs

    def _get_attrs(self):
        """
        :return: All the attributes on this object.
        """
        return {a for a in dir(self) if not (a.startswith('_') or callable(getattr(self, a)))}

    def _get_simple_value_type(self, value):
        """
        :param value: The value to type.
        :return: The ConfigType of the value (excluding CSV types)
        """
        if isinstance(value, bool):
            return ConfigType.Bool
        elif isinstance(value, int):
            return ConfigType.Int
        elif isinstance(value, float):
            return ConfigType.Float
        elif isinstance(value, str):
            return ConfigType.String
        else:
            return ConfigType.Unknown

    def _get_value_type(self, value):
        """
        :param value: The value to type
        :return: The type of the value including if it's a CSV or not.
        """
        t = self._get_simple_value_type(value)
        if t != ConfigType.Unknown:
            return t
        if isinstance(value, (list, tuple, set)):
            if len(value) > 0:
                return ConfigType.Csv | self._get_simple_value_type(value[0])
            return ConfigType.Csv

        return t

    def _attr_to_str(self, attr):
        """
        :param attr: The attribute to stringify
        :return: The attribute as a string
        """
        value = getattr(self, attr)
        if self._get_value_type(value) & ConfigType.Csv:
            return ", ".join([str(x) for x in value])
        else:
            return str(value)

    def _save_attr(self, section, attr):
        """
        Save the value of an attribute to a section.
        :param section: The section to save the attribute to
        :param attr: The attribute to save
        """
        section[attr] = self._attr_to_str(attr)

    def _load_csv_attr(self, section, t, attr):
        """
        Load the value of a CSV attribute
        :param section: The section to load from
        :param t: The ConfigType of the value
        :param attr: The attribute to load.
        """
        value = section.get(attr)
        value = [ConfigType.toValue(t, x.strip()) for x in value.split(",")]
        setattr(self, attr, value)

    def _load_attr(self, section, attr):
        """
        Load an attribute
        :param section: The section to load from
        :param attr: The attribute to load
        """
        value = getattr(self, attr)
        t = self._get_value_type(value)

        if t == ConfigType.Bool:
            value = section.getboolean(attr)
            setattr(self, attr, value)
        elif t == ConfigType.Int:
            value = section.getint(attr)
            setattr(self, attr, value)
        elif t == ConfigType.Float:
            value = section.getfloat(attr)
            setattr(self, attr, value)
        elif t == ConfigType.String:
            value = section.get(attr)
            setattr(self, attr, value)
        elif t & ConfigType.Csv:
            self._load_csv_attr(section, t, attr)

    def save(self, writeable):
        """
        Save config to an output stream/file.
        :param writeable: The file to write to
        """
        attrs = self._get_attrs()
        config = configparser.ConfigParser()

        for section_name, items in sorted(self._sections.items()):
            config[section_name] = OrderedDict()
            section = config[section_name]
            for attr in sorted(items):
                self._save_attr(section, attr)
                attrs.remove(attr)

        # anything not in an explicit section
        if len(attrs) > 0:
            config["other"] = OrderedDict()
            section = config["other"]
            for attr in sorted(attrs):
                self._save_attr(section, attr)

        if isinstance(writeable, str):
            with open(writeable, 'w') as configfile:
                config.write(configfile)
        else:
            config.write(writeable)

    def load(self, file_path):
        """
        Load config from a file
        :param file_path: The file to load from
        """
        attrs = self._get_attrs()
        config = configparser.ConfigParser()
        config.read(file_path)

        for section_name, items in self._sections.items():
            for attr in items:
                attrs.remove(attr)
                if config.has_option(section_name, attr):
                    self._load_attr(config[section_name], attr)

        section_name = "other"
        for attr in attrs:
            if config.has_option(section_name, attr):
                self._load_attr(config[section_name], attr)

        self.post_load()

    def post_load(self):
        """
        Called after load() completes. Use to fix-up variables if needed.
        """
        pass

    def uid(self):
        """
        Generate a unique ID for the config
        :return: A 16-byte unique ID for this object.
        """
        attrs = self._get_attrs()

        m = hashlib.sha256()
        for attr in sorted(attrs):
            if attr in self._blacklist:
                continue
            v = self._attr_to_str(attr).encode("utf-8")
            m.update(v)
        return m.hexdigest()[:16]
