# Python基础
## 魔法方法
* __new__: 从class建立一个object的过程;
    例如想要做一个singleton class（单例类），在建立一个class的object之前，判断一下有没有其他object已经被建立；还有需要用到metaclass有关的内容也会用到new；new有返回值，返回这个object；
* __init__: 有object之后给object初始化的过程;
    初始化使用 init；init没有返回值；
* __del__: 可以理解为析构函数，但不是析构函数;
    当对象被释放的时候需要干什么; `__del__`和python里面的关键字del没有关系，使用关键字del对象并不一定会触发`__del__`，del对象只是让这个对象少一个引用；只有当对象完全被释放的时候才会触发`__del__`
* __repr__: 返回object的字符串表示;
    返回更详细的信息，给开发者用的；需要显示调用(repr(obj))；`__repr__`找不到不会去找`__str__`;
* __str__:返回object的字符串表示;
    返回人类更容易理解的string，注重可读性；打印对象/字符串时被调用;`__str__`找不到会自动去找`__repr__`;
* __format__: 使用某种格式打印object；
    当使用 `f"{obj:格式}"` 或者
    ```
    "{}".format(obj)
    "{:格式}".format(obj)
    format(obj, "格式")
    ```
    的时候会被调用，如果找不到会去找`__str__`
* __bytes__：客制化object的bytes表示;
    bytes(obj)显示调用