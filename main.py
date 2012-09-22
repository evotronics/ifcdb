import hashlib
import json
import logging
import os
import random
import re
import sys
import urllib
import datetime

#sys.path[0:0] = ['lib']

import webapp2
from webapp2 import Route
from webapp2_extras import routes
from webapp2_extras import jinja2
from google.appengine.ext import ndb
from google.appengine.ext import blobstore
from google.appengine.ext.webapp import blobstore_handlers
from google.appengine.ext import deferred
from google.appengine.api import files
from mapreduce import control as mr_ctl
from mapreduce import operation as mr_op

logging.getLogger().setLevel(logging.DEBUG)

# prefix from 2009-10-08 - now
BASE_TAG_PREFIX = 'tag:ifcdb.com,2009:/1'
ID_TAG_PREFIX = BASE_TAG_PREFIX + '/ids'

def id_tag(local):
    return ID_TAG_PREFIX + '/' + local

class BaseHandler(webapp2.RequestHandler):
    @webapp2.cached_property
    def jinja2(self):
        return jinja2.get_jinja2(app=self.app)

    def render_template(self, filename, **template_args):
        self.response.write(self.jinja2.render_template(filename, **template_args))

    def render_json(self, data, debug=False):
        separators = (',',':')
        indent = None
        if debug:
            indent = 2
            separators = (', ',': ')
        self.response.content_type = 'application/json'
        self.response.write(json.dumps(data, indent=indent, separators=separators))

class ResourceHandler(BaseHandler):
    def __init__(self, *args, **kwds):
        super(ResourceHandler, self).__init__(*args, **kwds)
        self._type_map = {
            'text/html': self.get_html,
            'application/json': self.get_json,
        }
        self._types = self._type_map.keys()

    def get(self, *args, **kwds):
        ct = self.request.accept.best_match(self._types)
        get_type = self._type_map[ct]
        return get_type(*args, **kwds)

    def get_html(self, *args, **kwds):
        self.abort(500)

    def get_json(self, *args, **kwds):
        self.abort(500)

class IndexHandler(BaseHandler):
    def get(self):
        self.render_template('index.html', name=self.request.get('name'))

#jinja_environment = jinja2.Environment(
#        loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))

#class Configuration(db.Model):
#    some_config_data = db.StringProperty()
#
#    _config_cache = None
#    _config_lock = threading.Lock()
#
#    @classmethod
#    def get_config(cls):
#        with cls._config_lock:
#            if not cls._config_cache:
#                cls._config_cache = cls.get_by_key_name('config')
#        return cls._config_cache

class HomeHandler(BaseHandler):
    def get(self):
        #self.response.headers['Content-Type'] = 'text/plain'
        #self.response.out.write('Hello, webapp2 World!')

        fortunes = FC.query().order(-FC.meta.updated_index).fetch(20)
        args = {
            'fortunes': fortunes,
        }
        #self.render_template('index.html', name=self.request.get('name'))
        self.render_template('home.html', **args)

class FortunesHandler(BaseHandler):
    def get(self):
        vars = make_default_vars(self.request)
        self.response.out.write(render("fc-all.html", vars))

    def post(self):
        return # FIXME
        fc = FortuneCookie()
        fc.content = self.request.get("content")
        fc.put()
        logging.info("created")

class AddFortuneHandler(BaseHandler):
    def get(self):
        vars = make_default_vars(self.request)
        vars["use_forms"] = True
        self.response.out.write(render("fc-add.html", vars))

    def post(self):
        vars = make_default_vars(self.request)
        vars["preview"] = self.request.get("content")
        vars["content"] = self.request.get("content")
        logging.info("DBGc %s", self.request.get("new"))
        logging.info("DBGp %s", self.request.get("preivew"))
        self.response.out.write(render("fc-add.html", vars))

class UploadFortuneHandler(BaseHandler):
    def get(self):
        upload_url = blobstore.create_upload_url('/fc/-/upload_file')
        args = {
            'upload_url': upload_url,
        }
        self.render_template('fc-upload.html', **args)

    #def post(self):
    #    # FIXME: data upload API

class Id(ndb.Model):
    """Pseudo-class used for global allocate_ids()."""
    pass

class RandomIndexProperty(ndb.IntegerProperty):
    """A IntegerProperty which is initialized to a random value."""
    _invalid_index = -2**63
    def __init__(self, **kwds):
        kwds.setdefault('default', self._invalid_index)
        super(ndb.IntegerProperty, self).__init__(**kwds)

    def _to_base_type(self, value):
        if value == self._invalid_index:
            return random.randint(-2**63 + 1, 2**63-1)

slug_re = re.compile('^[a-z]([a-z0-9-]*[a-z0-9]$|$)')
class SlugProperty(ndb.StringProperty):
    """A SlugProperty holds a URL slug value."""
    def _validate(self, value):
        assert slug_re.match(value) is not None

class WeightProperty(ndb.FloatProperty):
    """A WeightProperty is a float value in [-1.0,1.0]."""
    def _validate(self, value):
        assert value >= -1.0 and value <= 1.0

#def _sort_seq(seq):
#    seqstr = str(seq)
#    seqlen = len(seqstr)
#    assert seqlen > 0 and seqlen <= 26
#    return chr(ord('a') + seqlen - 1) + seqstr

class VersionProperty(ndb.StringProperty):
    """A sortable dotted version string."""
    @staticmethod
    def _part_to_sortable(part):
        part_len = len(part)
        assert part_len > 0 and part_len <= 26
        return chr(ord('a') + part_len - 1) + part

    @staticmethod
    def _part_from_sortable(sortable):
        assert (len(sortable) - 1) == (ord(sortable[0]) - ord('a') + 1)
        return sortable[1:]
        
    @staticmethod
    def _to_sortable(value):
        parts = value.split('.')
        return '.'.join([VersionProperty._part_to_sortable(part) for part in parts])

    @staticmethod
    def _from_sortable(value):
        parts = value.split('.')
        return '.'.join([VersionProperty._part_from_sortable(part) for part in parts])

    def _to_base_type(self, value):
        return self._to_sortable(value)

    def _from_base_type(self, value):
        return self._from_sortable(value)

    #FIXME: add version inc code
    def add_version(self, inc):
        # inc = x.y.z, = = same, + = inc
        # only need to specify up to the +, rest go to 0
        # 1.2.3 + =.=.+ = 1.2.4
        # 1.2.3 + =.+ = 1.3.0
        # 1.2.3 + + = 2.0.0
        pass

def _sortable_index(value, id, sortable_version):
    return '%s|%s|%s' % (value, id, sortable_version)

def _published_index(self):
    #return sort_index(self.published, self.key.id(), self.seq)
    return sort_index(self.published, None, self.seq)

def _updated_index(self):
    #return sort_index(self.updated, self.key.id(), self.seq)
    return sort_index(self.updated, None, self.seq)

class Meta(ndb.Model):
    # model version
    model_version = ndb.IntegerProperty('v_', default=1, indexed=False)
    # log entry for this version
    log_entry = ndb.KeyProperty(indexed=False)
    # version, cached from log
    version = ndb.StringProperty(indexed=False)
    # set when created, cached from log
    created = ndb.DateTimeProperty(indexed=False)
    # set when first published, cached from log
    published = ndb.DateTimeProperty(indexed=False)
    # set when updated, cached from log
    updated = ndb.DateTimeProperty(indexed=False)
    # date + id + sortable_version
    published_index = ndb.StringProperty()
    updated_index = ndb.StringProperty()
    slug = SlugProperty(indexed=False)

    def sync(self, model, sortable_version):
        id = model.key.id()
        pass

#class Log(ndb.Model):
#    # parent = Model
#    pass

class LogContent(ndb.Model):
    # parent = LogEntry
    content = ndb.JsonProperty(required=True)

class LogEntry(ndb.Model):
    # parent = Loggable Model
    # model version
    model_version = ndb.IntegerProperty('v_', default=1, indexed=False)
    # date
    date = ndb.DateTimeProperty(indexed=False)
    # latest sequence version id as sorted semver 
    #version = VersionProperty(default='1', indexed=False)
    version = ndb.StringProperty(indexed=False)
    # date + parent entity id + sorted version
    date_index = ndb.StringProperty(required=True)
    # references to predecessor versions
    predecessors = ndb.KeyProperty(repeated=True, kind='LogEntry', indexed=False)
    # reference to author account
    #FIXME
    #author = ndb.KeyProperty(required=True)
    author = ndb.KeyProperty()
    # summary, atom text content
    summary = ndb.JsonProperty(required=True)
    # full entry content, external to avoid loading cost
    content = ndb.KeyProperty(required=True, kind=LogContent)

class Loggable(object):
    """Mixin to add logging behavior to a Model."""
    @ndb.transactional
    def _logged_put(self, entry_id, content_id, version_inc='+'):
        # make new entry
        has_prev = self.meta.log_entry is not None
        prev = self.meta.log_entry.get() if has_prev else None
        date = datetime.datetime.utcnow()
        if has_prev:
            # FIXME version.add_version(version_inc)
            version = str(int(self.meta.version) + 1)
        else:
            # FIXME better default
            version = '1'
        sortable_version = VersionProperty._to_sortable(version)
        date_index = _sortable_index(str(date), self.key.id(), sortable_version)
        entry_key = ndb.Key(LogEntry, entry_id, parent=self.key)
        content_key = ndb.Key(LogContent, content_id, parent=entry_key)
        predecessors = [self.meta.log_entry] if self.meta.log_entry else []
        entry = LogEntry(
                key=entry_key,
                date=date,
                version=version,
                date_index=date_index,
                predecessors=predecessors,
                author=None,
                summary='',
                content=content_key)
        # FIXME build log content
        content = LogContent(key=content_key, content={})

        # update current entry
        self.meta.log_entry = entry.key
        self.meta.version = version
        self.meta.updated = date

        # put it all
        ndb.put_multi([self, entry, content])

    def logged_put(self, version_inc="+"):
        # FIXME
        entry_id, content_id = Id.allocate_ids(2)
        self._logged_put(str(entry_id), str(content_id), version_inc)

class Label(ndb.Model, Loggable):
    meta = ndb.StructuredProperty(Meta)
    label = ndb.StringProperty(indexed=False)
    description = ndb.TextProperty()
    # alternative text for search term generation
    #alt = 
    # search terms
    #search =

class Tag(ndb.Model, Loggable):
    label = ndb.KeyProperty(required=True, kind=Label)
    weight = WeightProperty(default=1.0)

class Relation(ndb.Model, Loggable):
    """Link relation between a resource and a target."""
    meta = ndb.StructuredProperty(Meta)
    label = ndb.StringProperty(indexed=False)
    pass

# http://tools.ietf.org/html/rfc5988
# http://www.iana.org/assignments/link-relations/link-relations.xml
class Link(ndb.Model, Loggable):
    rel = ndb.KeyProperty(required=True, kind=Relation)
    weight = WeightProperty(default=1.0)
    target = ndb.KeyProperty(required=True)

class TagsHandler(BaseHandler):
    def get(self):
        tags = Tag.query().order(-Tag.meta.created).fetch(20)
        args = {
            'tags': tags,
        }
        self.render_template('tags-all.html', **args)

    def post(self):
        first, last = Id.allocate_ids(1)
        t = Tag(id=last)
        t.slug = self.request.get("slug")
        t.label = self.request.get("label")
        t.meta = Meta()
        t.put()
        logging.info("tag created")

class AddTagHandler(BaseHandler):
    def get(self):
        args = {}
        self.render_template('tags-add.html', **args)

class TagHandler(BaseHandler):
    def get(self, tag_id):
        tag_id = int(tag_id)
        key = ndb.Key('Tag', tag_id)
        tag = key.get()
        fortunes = FC.query(FC.tags == tag_id) \
                   .order(FC.meta.created) \
                   .fetch(20)
        args = {
            'tag': tag,
            'fortunes': fortunes,
        }
        self.render_template('tags-tag.html', **args)

    def post(self):
        return # FIXME
        t = Tag()
        t.slug = self.request.get("slug")
        t.label = self.request.get("label")
        t.put()
        logging.info("tag created")

class FC(ndb.Model, Loggable):
    meta = ndb.StructuredProperty(Meta)
    random = RandomIndexProperty()
    # JSON-LD Atom Entry
    entry = ndb.JsonProperty(default={})
    enabled = ndb.BooleanProperty(default=True)
    tags = ndb.StructuredProperty(Tag, repeated=True)

class HashScheme(ndb.Model, Loggable):
    meta = ndb.StructuredProperty(Meta)
    # id is short name, label is longer readable text
    label = ndb.TextProperty()
    # description of the conversion method
    description = ndb.TextProperty()

class FCHash(ndb.Model):
    """
    Hash of FC content.
    Hashes added for each content change.
    Parent is LogEntry.
    ID is "{method id}:{hash}" for small and fast key lookups.
    """
    pass

def fc_content_hashes(content):
    # h-1
    h_1 = hashlib.sha1()
    h_1.update(content)
    h_1 = 'h-1:' + h_1.hexdigest()

    # h-2
    h_2 = hashlib.sha1()
    # FIXME norm whitespace
    h_2.update(content.lower())
    h_2 = 'h-2:' + h_2.hexdigest()

    # h-3
    h_3 = hashlib.sha1()
    # FIXME norm punc, norm whitespace
    h_3.update(content.lower())
    h_3 = 'h-3:' + h_3.hexdigest()

    return [h_1, h_2, h_3]

def fc_from_json(json_str):
    data = json.loads(json_str)
    fc = FC(id=data['id'], content=data['c'], tags=[], meta=Meta())
    return fc

def import_fc_if_new(fc):
    hashes = fc_content_hashes(fc.content)
    # FIXME: add logentry(p=fc), hash(p=le)
    # get le id
    # make le w p=fc
    # FIXME: change parent to le
    fc_hashes = [FCHash(id=hash, parent=fc.key) for hash in hashes]
    fc_hash_keys = [hash.key for hash in fc_hashes]
    fcs = ndb.get_multi(fc_hash_keys)
    if not any(fcs):
        # FIXME: add le
        ndb.put_multi([fc] + fc_hashes)
    else:
        # FIXME: try to unallocate all ids
        pass

def mr_import_fc_line((byte_offset, json_line)):
    logging.info('mr_import_fc_line b:%s' % (json_line))
    fc = fc_from_json(json_line)
    # no ndb support yet
    #yield mr_op.db.Put(fc)
    import_fc_if_new(fc)
    yield mr_op.counters.Increment('fortunes')

def mr_import_fc_file(blob_key):
    logging.info('mr_import_fc_file start bk:%s' % (blob_key))
    mr_ctl.start_map(
        name='Import a fortune file',
        handler_spec='main.mr_import_fc_line',
        reader_spec='mapreduce.input_readers.BlobstoreLineInputReader',
        mapper_parameters={
            # FIXME add blob key to done callback
            'done_callback': '/mr_done',
            'blob_keys': str(blob_key),
        },
    )
    logging.info('mr_import_fc_file end')

def import_fc_file(blob_key):
    logging.info('Start import blob_key: %s' % (blob_key))
    r = blobstore.BlobReader(blob_key)
    #for line in r:
    #    logging.info('IMPORT: %s' % (line))
    # content lines
    # FIXME: batch and/or use async
    cnt = 0
    for line in r:
        logging.info('Import line: %s' % (line))
        fc = fc_from_json(line)
        ndb.put_multi(fc)
        cnt += 1
    r.close()
    logging.info('Done import blob_key: %s cnt:%d' % (blob_key, cnt))

def transform_to_mappable_file(in_key):
    """Transforms fortune style file to single line JSON-LD file.
    Output blob is suitable for mapreduce.
    """
    logging.info('Transforming blob_key:%s' % (in_key))
    r = blobstore.BlobReader(in_key)

    # create list of records
    # content lines
    cl = []
    records = []
    for line in r:
        if line.strip() == '%' and len(cl) > 0:
            # create content
            c = ''.join(cl)
            # clear lines
            cl = []
            record = {
                'c': c
            }
            records.append(record)
        else:
            cl.append(line)
    r.close()

    out_key = None
    if records:
        # Create the file
        wfn = files.blobstore.create(mime_type='text/plain')
        # Open the file and write to it
        with files.open(wfn, 'a') as w:
            # Allocate ids
            start, end = Id.allocate_ids(len(records))
            for id, r in enumerate(records, start):
                record['id'] = str(id)
                s = json.dumps(record, separators=(',', ':')) + '\n'
                # FIXME write to iostring first?
                w.write(s)

        # Finalize the file. Do this before attempting to read it.
        files.finalize(wfn)

        # Get the file's blob key
        out_key = files.blobstore.get_blob_key(wfn)

        #deferred.defer(import_fc_file, out_key)
        deferred.defer(mr_import_fc_file, out_key)

    logging.info('Transformed blob_key:%s to blob_key:%s cnt:%d' % (in_key, out_key, len(records)))

    blobstore.delete(in_key)

class UploadFileFortuneHandler(blobstore_handlers.BlobstoreUploadHandler):
    def post(self):
        # 'file' is file upload field in the form
        upload_files = self.get_uploads('file')
        blob_info = upload_files[0]
        self.redirect('/fc/-/upload/%s' % blob_info.key())
        deferred.defer(transform_to_mappable_file, blob_info.key())

# FIXME: process data
# - create event id
# - create transform task 
# - add task id, owner to event
# - put event
# - start task
# - task
#   - blob reader => xform => blob writer
#   - update task
#   - update event
#   - start mapper
# - add mapper
#   - v1 check basic hash and add raw data + hash(es) as new fc
#   - v2 do initial formatting
# - format mapper 
#   - run various new formatters on fcs

class UploadStatusFortuneHandler(BaseHandler):
    def get(self, uid):
        resource = str(urllib.unquote(uid))
        blob_info = blobstore.BlobInfo.get(uid)
        #self.send_blob(blob_info)
        args = {
            'blob_info': blob_info,
        }
        self.render_template('fc-status.html', **args)

class FortuneFeedHandler(BaseHandler):
    def get(self, special=None):
        self.response.out.write("FIXME")

class EditFortuneHandler(BaseHandler):
    def get(self):
        vars = make_default_vars(self.request)
        self.response.out.write(render("fc-edit.html", vars))

class FortuneHandler(ResourceHandler):
    def get_html(self, fc):
        #fc = FortuneCookie.get_by_key_name(FC_KEY_NAME_PREFIX + key_name)
        fortune = FC.get_by_id(fc)
        log_entry = fortune.meta.log_entry.get()
        log_entries = LogEntry.query(ancestor=fortune.key).order(-LogEntry.date_index).fetch()
        args = {
            'fc': fortune,
            'le': log_entry,
            'les': log_entries,
        }
        self.render_template('fc-fc.html', **args)

    def get_json(self, fc):
        #fc = FortuneCookie.get_by_key_name(FC_KEY_NAME_PREFIX + key_name)
        fortune = FC.get_by_id(fc)
        log_entry = fortune.meta.log_entry.get()
        log_entries = LogEntry.query(ancestor=fortune.key).order(-LogEntry.date_index).fetch()
        data = {
            '@context': 'IFCDBCTX',
            '@id': '/fc/' + fortune.key.id(),
            '@type': 'FortuneCookie',
            'published': '',
            'updated': '',
            #'le': log_entry,
            #'les': log_entries,
        }
        self.render_json(data)

class TasksHandler(BaseHandler):
    def get(self):
        self.render_template('tasks.html')

class TaskHandler(BaseHandler):
    def get(self, event_id):
        args = {
            'event_id': event_id
        }
        self.render_template('task.html', **args)

class InfoHandler(BaseHandler):
    def get(self, page):
        self.render_template('%s.html' % (page))

class HelpHandler(BaseHandler):
    def get(self):
        self.render_template('help.html')

class IfcdbConfig(ndb.Model):
    pass

def admin_init():
    # WARNING: not safe to run in parallel!
    # This is rare code to run so just assuming it will be done by hand
    # and simple code and pass/fail reporting is sufficient.

    # HashScheme
    HashScheme.logged_add(
            id='h-1',
            slug='h-1',
            label='Hash Scheme 1',
            description='content|sha1|base16')
    HashScheme.logged_add(
            id='h-2',
            slug='h-2',
            label='Hash Scheme 2',
            description='content|lower|norm-ws|sha1|base16')
    HashScheme.logged_add(
            id='h-1',
            slug='h-1',
            label='Hash Scheme 3',
            description='content|lower|norm-punc|norm-ws|sha1|base16')

    # Relations
    Relation.get_or_insert('tag',
            label='Hash Scheme 2',
            description='content|lower|norm-ws|sha1|base16')
    Relation.get_or_insert('tag',
        )

    # Admin Account

    return True

class AdminInitHandler(BaseHandler):
    def get(self):
        self.render_template('admin-init.html')
    def post(self):
        success = admin_init()

class TestPing(BaseHandler):
    def get(self):
        pass

class Test1(BaseHandler):
    def get(self):
        n = 3
        start, end = Id.allocate_ids(n)
        for id in range(start, end + 1):
            id = str(id)
            fc = FC(id=id, meta=Meta())
            fc.put()
        self.render_template('t1.html')

class Test2Model(ndb.Model, Loggable):
    meta = ndb.StructuredProperty(Meta)
    txt = ndb.StringProperty()

class Test2(BaseHandler):
    def get(self):
        fc = FC.get_or_insert('fc1', meta=Meta())
        fc.entry['@type'] = 'Entry'
        fc.entry['content'] = {}
        fc.entry['content']['@type'] = 'Content'
        fc.entry['content']['type'] = 'text/plain'
        fc.entry['content']['body'] = 'test 0'
        fc.logged_put()
        fc.entry['content']['body'] = 'test 1'
        fc.logged_put()
        fc.entry['content']['body'] = 'test 2'
        fc.logged_put()
        fc.entry['content']['body'] = 'test 3'
        fc.logged_put()

class Test3(ResourceHandler):
    def get_html(self):
        logging.info('t3h')
        self.response.write('<html><body>moo</body></html>')
    def get_json(self):
        logging.info('t3j')
        self.render_json({"json":True})

routes = [
        Route(r'/', handler='main.HomeHandler'),
        routes.PathPrefixRoute(r'/fc', [
            Route(r'/', handler='main.FortunesHandler'),
            Route(r'/-/add', handler='main.AddFortuneHandler'),
            Route(r'/-/upload', handler='main.UploadFortuneHandler'),
            Route(r'/-/upload_file', handler='main.UploadFileFortuneHandler'),
            Route(r'/-/upload/<uid>', handler='main.UploadStatusFortuneHandler'),
            #Route(r'/-/feed', handler='main.FortuneFeedHandler'),
            #Route(r'/-/feed/<feedtype>', handler='main.FortuneFeedHandler'),
            Route(r'/<fc>/-/edit', handler='main.EditFortuneHandler'),
            Route(r'/<fc>/<slug>', handler='main.FortuneHandler'),
            Route(r'/<fc>', handler='main.FortuneHandler'),
        ]),
        routes.PathPrefixRoute(r'/tags', [
            Route(r'/', handler='main.TagsHandler'),
            Route(r'/-/add', handler='main.AddTagHandler'),
            Route(r'/<tag>', handler='main.TagHandler'),
        ]),
        routes.PathPrefixRoute(r'/labels', [
            Route(r'/', handler='main.LabelsHandler'),
            Route(r'/-/add', handler='main.AddLabelHandler'),
            Route(r'/<label>', handler='main.LabelHandler'),
        ]),
        routes.PathPrefixRoute(r'/admin', [
            routes.PathPrefixRoute(r'/-', [
                Route(r'/init', handler='main.AdminInitHandler'),
            ]),
        ]),
        Route(r'/sys/tasks', handler='main.TasksHandler'),
        Route(r'/sys/tasks/<task>', handler='main.TaskHandler'),
        #Route(r'/sys/log', LogsHandler),
        #Route(r'/sys/log/(.*)', LogHandler),
        Route(r'/(about|contact|legal|privacy|terms)', 'main.InfoHandler'),
        Route(r'/help', 'main.HelpHandler'),
        routes.PathPrefixRoute(r'/tests', [
            #Route(r'/', handler=''),
            Route(r'/ping', handler='main.TestPing'),
            Route(r'/t1', handler='main.Test1'),
            Route(r'/t2', handler='main.Test2'),
            Route(r'/t3', handler='main.Test3'),
        ]),
]
debug = os.environ.get('SERVER_SOFTWARE', '').startswith('Dev')
config = {}
config['webapp2_extras.jinja2'] = {
    'template_path': ''
}

app = ndb.toplevel(webapp2.WSGIApplication(
            routes=routes, debug=debug, config=config))
