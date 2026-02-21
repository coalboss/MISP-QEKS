import traceback
import importlib

def get_module(module_name):
    items = module_name.split('.')
    m_name = items[0].strip()
    for i in range(1,len(items)-1):
        m_name = m_name + '.' + items[i].strip()
    classname = items[-1].strip()
    m = importlib.import_module(m_name)
    module = getattr(m, classname)
    return module

def printer(func):
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return 1
        except Exception:
            print(traceback.format_exc())
            return 0
    return handler