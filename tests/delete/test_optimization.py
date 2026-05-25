"""
Regression tests for #30191 — Optimize .delete() to use only required fields.

During QuerySet.delete(), Collector.related_objects() now applies .only() so
that only the pk (and any fields targeted by to_field FKs from further
cascade relations) are SELECTed when fetching related objects, as long as no
pre_delete/post_delete signals are registered for the model being collected.

This prevents crashes caused by fetching large or corrupt columns (e.g. a
TextField containing invalid UTF-8 bytes from old Python-2 data) even though
those columns are completely irrelevant to the delete operation.
"""

from django.db import models
from django.db.models.deletion import Collector, get_candidate_relations_to_delete
from django.test import TestCase
from django.test.utils import isolate_apps

from .models import (
    A, Avatar, R, RChild, S, T, U, User,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_related(parent_model, child_model):
    """Return the first ForeignObjectRel from child_model pointing to parent_model."""
    return next(
        rel for rel in get_candidate_relations_to_delete(parent_model._meta)
        if rel.related_model is child_model
    )


def _deferred_loading(qs):
    """Return (field_names_frozenset, include_only_bool) for the queryset."""
    return qs.query.deferred_loading


# ---------------------------------------------------------------------------
# Unit tests: Collector.related_objects() queryset shape
# ---------------------------------------------------------------------------

class RelatedObjectsFieldRestrictionTests(TestCase):
    """
    Unit tests that check the deferred_loading of the QuerySet returned by
    Collector.related_objects() — without hitting the database at all.

    deferred_loading is a tuple (frozenset_of_names, include_only_bool).
    When include_only_bool=False, only the named fields are loaded ("whitelist").
    When include_only_bool=True, the named fields are deferred ("blacklist");
    an empty frozenset means nothing is deferred, i.e. all fields are loaded.
    """

    # -- no signals: only pk (and FK-target fields) selected -----------------

    def test_only_pk_selected_when_no_signals(self):
        """
        Without any delete signals, related_objects() uses .only(pk) so that
        unneeded columns (TextFields, etc.) are never fetched.

        S has fields: id (pk), r_id.
        T→S FK targets S.id, so rhs_field = S.id = pk.
        Result: field_names = {'id'}, no extra columns.
        """
        collector = Collector(using='default')
        related = _get_related(R, S)
        r = R(pk=1)

        qs = collector.related_objects(related, [r])

        field_names, include_only = _deferred_loading(qs)
        self.assertIs(include_only, False)
        # S._meta.pk.name = 'id'. Only 'id' needed.
        self.assertEqual(field_names, frozenset(['id']))

    def test_only_pk_selected_for_leaf_model(self):
        """
        A leaf model (nothing cascades from it) also returns only its pk.
        U has fields: id (pk), t_id.  Nothing points to U.
        """
        collector = Collector(using='default')
        related = _get_related(T, U)
        t = T(pk=1)

        qs = collector.related_objects(related, [t])

        field_names, include_only = _deferred_loading(qs)
        self.assertIs(include_only, False)
        self.assertEqual(field_names, frozenset(['id']))

    # -- signals: all fields must remain available ---------------------------

    def test_all_fields_with_pre_delete_signal(self):
        """
        When a pre_delete signal is registered for the model, .only() is NOT
        applied — signal handlers may read any field of the instance.
        """
        collector = Collector(using='default')
        related = _get_related(R, S)
        r = R(pk=1)

        def noop(*args, **kwargs):
            pass

        models.signals.pre_delete.connect(noop, sender=S)
        try:
            qs = collector.related_objects(related, [r])
            field_names, defer_mode = _deferred_loading(qs)
            # Unrestricted queryset: (frozenset(), True) = defer nothing = all fields.
            self.assertIs(defer_mode, True)
            self.assertEqual(field_names, frozenset())
        finally:
            models.signals.pre_delete.disconnect(noop, sender=S)

    def test_all_fields_with_post_delete_signal(self):
        """
        When a post_delete signal is registered for the model, .only() is NOT
        applied.
        """
        collector = Collector(using='default')
        related = _get_related(R, S)
        r = R(pk=1)

        def noop(*args, **kwargs):
            pass

        models.signals.post_delete.connect(noop, sender=S)
        try:
            qs = collector.related_objects(related, [r])
            field_names, defer_mode = _deferred_loading(qs)
            self.assertIs(defer_mode, True)
            self.assertEqual(field_names, frozenset())
        finally:
            models.signals.post_delete.disconnect(noop, sender=S)

    def test_all_fields_with_global_signal(self):
        """
        A signal connected without a specific sender (fires for all models)
        also prevents field restriction for any model.
        """
        collector = Collector(using='default')
        related = _get_related(R, S)
        r = R(pk=1)

        def noop(*args, **kwargs):
            pass

        models.signals.pre_delete.connect(noop)   # no sender = every model
        try:
            qs = collector.related_objects(related, [r])
            field_names, defer_mode = _deferred_loading(qs)
            self.assertIs(defer_mode, True)
            self.assertEqual(field_names, frozenset())
        finally:
            models.signals.pre_delete.disconnect(noop)

    # -- FK with to_field: non-pk target must be included --------------------

    @isolate_apps('delete')
    def test_to_field_target_included(self):
        """
        When model M has a unique field 'uuid' and child model C declares
        FK(M, to_field='uuid'), M must include 'uuid' in .only() when M is
        collected as a cascade child of some root — so that the subsequent
        filter C.objects.filter(m__in=m_instances) can use m.uuid as the
        IN-list value.

        Layout:  RootModel <- NodeModel (regular FK) <- EdgeModel (FK to_field='uuid')

        The call under test is related_objects(root→node relation, [root]),
        which fetches NodeModel instances.  field_names must include 'uuid'
        because EdgeModel.node targets NodeModel.uuid.
        """
        class RootModel(models.Model):
            class Meta:
                app_label = 'delete'

        class NodeModel(models.Model):
            root = models.ForeignKey(RootModel, models.CASCADE)
            uuid = models.CharField(max_length=36, unique=True)
            extra = models.TextField()   # must NOT appear in .only()

            class Meta:
                app_label = 'delete'

        class EdgeModel(models.Model):
            node = models.ForeignKey(NodeModel, models.CASCADE, to_field='uuid')

            class Meta:
                app_label = 'delete'

        # Expire reverse-relation caches so isolate_apps FK registrations are visible.
        RootModel._meta._expire_cache(forward=False)
        NodeModel._meta._expire_cache(forward=False)

        collector = Collector(using='default')

        # Find the reverse relation on RootModel that leads to NodeModel.
        candidate_rels = list(get_candidate_relations_to_delete(RootModel._meta))
        node_rels = [r for r in candidate_rels if r.related_model is NodeModel]
        self.assertEqual(
            len(node_rels), 1,
            "NodeModel.root reverse relation not found in RootModel._meta — "
            "FK lazy operation may not have run.",
        )

        related = node_rels[0]
        root = RootModel.__new__(RootModel)
        root.pk = 1
        # This fetches NodeModel instances (not EdgeModel).
        qs = collector.related_objects(related, [root])

        field_names, include_only = _deferred_loading(qs)
        self.assertIs(include_only, False)
        # NodeModel.id (pk) always included.
        self.assertIn('id', field_names)
        # NodeModel.uuid must be included: EdgeModel.node has to_field='uuid'.
        self.assertIn('uuid', field_names)
        # Irrelevant columns must NOT be included — this is the bug fix.
        self.assertNotIn('extra', field_names)
        self.assertNotIn('root_id', field_names)

    # -- MTI: pk is the parent link + inherited relations add parent-pk ------

    def test_mti_pk_is_parent_link(self):
        """
        For multi-table-inheritance models, the pk IS the parent link
        (RChild._meta.pk.name == 'r_ptr').

        .only('r_ptr') covers the parent-link column.  Additionally,
        get_candidate_relations_to_delete(RChild._meta) includes relations
        inherited from R (e.g. S.r → R.id), so 'id' also appears in
        field_names — this is correct: Django needs R.id to filter those
        inherited relations.  Non-pk fields of R (like 'is_default') must NOT
        appear.
        """
        collector = Collector(using='default')
        related = next(
            rel for rel in get_candidate_relations_to_delete(R._meta)
            if rel.related_model is RChild
        )
        r = R(pk=1)
        qs = collector.related_objects(related, [r])

        field_names, include_only = _deferred_loading(qs)
        self.assertIs(include_only, False)
        # The pk ('r_ptr') is always included.
        self.assertEqual(RChild._meta.pk.name, 'r_ptr')
        self.assertIn('r_ptr', field_names)
        # R.id is also included because S.r → R.id relation is inherited.
        self.assertIn('id', field_names)
        # Non-pk, non-FK-target fields are NOT included.
        self.assertNotIn('is_default', field_names)


# ---------------------------------------------------------------------------
# Integration tests: correctness with the optimization active
# ---------------------------------------------------------------------------

class DeleteCascadeCorrectnessTests(TestCase):
    """
    Full delete cascade must still produce correct results even though related
    objects are now loaded with fewer fields.
    """

    def test_cascade_deletes_all_related_objects(self):
        r = R.objects.create()
        s = S.objects.create(r=r)
        T.objects.create(s=s)
        T.objects.create(s=s)

        r.delete()

        self.assertFalse(R.objects.filter(pk=r.pk).exists())
        self.assertFalse(S.objects.filter(pk=s.pk).exists())
        self.assertFalse(T.objects.exists())

    def test_three_level_cascade(self):
        """R → S → T → U are all correctly deleted."""
        r = R.objects.create()
        s = S.objects.create(r=r)
        t = T.objects.create(s=s)
        u = U.objects.create(t=t)

        r.delete()

        for model, pk in [(R, r.pk), (S, s.pk), (T, t.pk), (U, u.pk)]:
            self.assertFalse(
                model.objects.filter(pk=pk).exists(),
                msg=f"{model.__name__} pk={pk} was not deleted",
            )

    def test_delete_returns_correct_counts(self):
        r = R.objects.create()
        s1 = S.objects.create(r=r)
        s2 = S.objects.create(r=r)
        T.objects.create(s=s1)
        T.objects.create(s=s2)

        _, counts = r.delete()

        self.assertEqual(counts['delete.S'], 2)
        self.assertEqual(counts['delete.T'], 2)

    def test_queryset_delete_cascade(self):
        r1 = R.objects.create()
        r2 = R.objects.create()
        S.objects.create(r=r1)
        S.objects.create(r=r2)

        deleted, _ = R.objects.all().delete()

        # 2 R + 2 S
        self.assertFalse(R.objects.exists())
        self.assertFalse(S.objects.exists())

    def test_mti_delete_cascade(self):
        """Deleting an RChild still removes the parent R row."""
        rc = RChild.objects.create()
        r_pk = rc.r_ptr_id

        rc.delete()

        self.assertFalse(RChild.objects.filter(pk=r_pk).exists())
        self.assertFalse(R.objects.filter(pk=r_pk).exists())


# ---------------------------------------------------------------------------
# Signal edge cases
# ---------------------------------------------------------------------------

class SignalEdgeCaseTests(TestCase):
    """
    When delete signals are connected, collected instances must expose all
    their fields to signal handlers without raising DeferredAttribute.
    """

    def test_post_delete_signal_can_read_non_pk_field(self):
        """
        A post_delete handler that reads a non-pk field (r_id) must not raise
        DeferredAttribute.  The signal fires inside the transaction, before
        the pk is set to None, so instance.pk is still the original value.
        """
        received = []

        def handler(sender, instance, **kwargs):
            # r_id is a non-pk field; accessing it proves the field is
            # NOT deferred when a signal listener is registered for S.
            received.append((instance.pk, instance.r_id))

        models.signals.post_delete.connect(handler, sender=S)
        try:
            r = R.objects.create()
            S.objects.create(r=r)
            r.delete()
        finally:
            models.signals.post_delete.disconnect(handler, sender=S)

        self.assertEqual(len(received), 1)
        pk, r_id = received[0]
        # post_delete fires inside the transaction; pk is not yet None.
        self.assertIsNotNone(pk)
        # r_id must be accessible (not deferred / not triggering a DB error
        # because the row is already deleted and the field was deferred).
        self.assertIsNotNone(r_id)

    def test_pre_delete_signal_can_read_non_pk_field(self):
        """Same guarantee for pre_delete — and here the row still exists in DB."""
        received_r_ids = []

        def handler(sender, instance, **kwargs):
            received_r_ids.append(instance.r_id)

        models.signals.pre_delete.connect(handler, sender=S)
        try:
            r = R.objects.create()
            S.objects.create(r=r)
            r.delete()
        finally:
            models.signals.pre_delete.disconnect(handler, sender=S)

        self.assertEqual(len(received_r_ids), 1)
        self.assertIsNotNone(received_r_ids[0])

    def test_signal_on_unrelated_model_does_not_affect_target(self):
        """
        A signal on model R must not prevent field restriction for model S.
        Only signals on the model being *collected* (S) matter.
        """
        def noop(*args, **kwargs):
            pass

        models.signals.pre_delete.connect(noop, sender=R)
        try:
            collector = Collector(using='default')
            related = _get_related(R, S)
            r = R(pk=1)
            qs = collector.related_objects(related, [r])
            # Signal is on R, not S → S's queryset is still restricted.
            field_names, include_only = _deferred_loading(qs)
            self.assertIs(include_only, False)
            self.assertEqual(field_names, frozenset(['id']))
        finally:
            models.signals.pre_delete.disconnect(noop, sender=R)


# ---------------------------------------------------------------------------
# Fast delete: .only() must not interfere with _raw_delete path
# ---------------------------------------------------------------------------

class FastDeleteOptimizationTests(TestCase):
    """
    Fast-deletable querysets use _raw_delete() (a direct DELETE SQL) and
    never iterate the queryset in Python — so .only() has no meaningful
    effect on them, but must not break them either.
    """

    def test_fast_delete_still_removes_rows(self):
        """
        A fast-deletable queryset with .only() applied (by related_objects)
        is still correctly deleted via _raw_delete().
        User.avatar FK → Avatar with CASCADE; User has no further cascades
        so User is always fast-deleted; Avatar.desc is never read into Python.
        """
        avatar = Avatar.objects.create(desc='some text')
        user = User.objects.create(avatar=avatar)

        avatar.delete()

        self.assertFalse(User.objects.filter(pk=user.pk).exists())
        self.assertFalse(Avatar.objects.filter(pk=avatar.pk).exists())

    def test_fast_delete_does_not_load_objects_into_python(self):
        """
        Fast-deletable related models must NOT be loaded into Python.
        Their .only() queryset must go through _raw_delete, not __iter__.
        2 queries: 1 fast-DELETE User, 1 DELETE Avatar.
        """
        avatar = Avatar.objects.create(desc='never read')
        User.objects.create(avatar=avatar)

        with self.assertNumQueries(2):
            avatar.delete()


# ---------------------------------------------------------------------------
# Query count: optimization must not add extra queries
# ---------------------------------------------------------------------------

class QueryCountTests(TestCase):
    """
    The field restriction changes WHICH columns are fetched but must not
    change HOW MANY queries are executed.
    """

    def test_s_delete_query_count(self):
        """
        Deleting an S with 1 cascading T (and no U rows) must execute exactly
        4 queries — same as before the optimization:

          1. SELECT T.id  (related objects, now .only('id') instead of *)
          2. fast-DELETE U (0 rows, but the queryset is still executed)
          3. DELETE T
          4. DELETE S
        """
        s = S.objects.create(r=R.objects.create())
        T.objects.create(s=s)

        with self.assertNumQueries(4):
            s.delete()

    def test_bulk_s_delete_query_count(self):
        """
        test_bulk equivalent: 2×GET_ITERATOR_CHUNK_SIZE T rows.
        Query count must match the pre-optimization value.

          1    SELECT T.id  (one batch: 200 < SQLite bulk_batch_size=500)
          1    fast-DELETE U (0 rows)
          2    DELETE T     (2 batches of GET_ITERATOR_CHUNK_SIZE=100)
          1    DELETE S
        = 5 queries total
        """
        from django.db.models.sql.constants import GET_ITERATOR_CHUNK_SIZE
        s = S.objects.create(r=R.objects.create())
        for _ in range(2 * GET_ITERATOR_CHUNK_SIZE):
            T.objects.create(s=s)

        with self.assertNumQueries(5):
            s.delete()
